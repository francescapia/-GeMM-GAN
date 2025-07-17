import os
from tqdm import tqdm
from pathlib import Path
import argparse
from datetime import datetime
import numpy as np
from sklearn.decomposition import PCA 

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn import MultiheadAttention

from generative_model_utils import *
from unsupervised_metrics import *
from corr_score import * 
from visualization import *
from multi_patch_multi_token_gan_dataloader import *
from utility_evaluation import UtilityEvaluator
from utility_primary_s_evaluation import UtilityEvaluatorPrimary
from privacy_evaluator import dcr, nndr, retrieval_accuracy
from glob import glob

import warnings
warnings.filterwarnings("ignore")


def save_numpy(file, data):
    with open(file, 'wb') as f:
        np.save(f, data)

def  wasserstein_loss(y_pred, y_true):
    #  distance function defined between probability distributions on a given metric space M
    return(torch.mean(y_pred * y_true))

def G_loss(fake_labels):
    # generator loss
    # fake labels -> all fake samples are labeled as true samples
    return wasserstein_loss(fake_labels, -torch.ones_like(fake_labels))

def D_loss(real_labels, fake_labels):

    loss_real = wasserstein_loss(-torch.ones_like(real_labels), real_labels)
    loss_fake = wasserstein_loss(torch.ones_like(fake_labels), fake_labels)
    total_loss = loss_real + loss_fake
    return total_loss, loss_real, loss_fake

def contrastive_loss(a, b, temperature=0.1) -> Tensor:
    a = F.normalize(a, dim=-1)
    b = F.normalize(b, dim=-1)
    logits = torch.matmul(a, b.T) / temperature
    labels = torch.arange(len(a)).to(a.device)
    return F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)


def build_linear_block(input_dims, output_dims, negative_slope = 0.0, is_bn= False):
    '''Paramters:
            -input_dims
            - output_dims
            - negative_slope: defeault 0.0 -> standard ReLU
            - is_bn -> batch normalization
    '''
    if is_bn:
        net = nn.Sequential(
            nn.Linear(input_dims, output_dims),
            nn.BatchNorm1d(output_dims),
            nn.LeakyReLU(negative_slope=negative_slope)
        )
    else: 
        net = nn.Sequential(
            nn.Linear(input_dims, output_dims),
            nn.LeakyReLU(negative_slope=negative_slope))
            #nn.ReLU())
    return net

def build_generator(input_dims, generator_dims, negative_slope = 0.0, is_bn = False):

    generator = nn.ModuleList()
    for i in range(len(generator_dims)):
        if i == 0:
            print(input_dims)
            generator.append(build_linear_block(input_dims, generator_dims[i], negative_slope=negative_slope, is_bn= is_bn))
        else:
            generator.append(build_linear_block(generator_dims[i-1], generator_dims[i], negative_slope=negative_slope, is_bn= is_bn))
    return generator

def build_discriminator(input_dims, dicriminator_dims, negative_slope= 0.0, is_bn=False):

    dicriminator = nn.ModuleList()
    for i in range(len(dicriminator_dims)):
        if i == 0:
            dicriminator.append(build_linear_block(input_dims, dicriminator_dims[i], negative_slope=negative_slope, is_bn=is_bn))
        else:
            dicriminator.append(build_linear_block(dicriminator_dims[i-1], dicriminator_dims[i], negative_slope=negative_slope, is_bn=is_bn))
    return dicriminator

class generator(nn.Module):

    def __init__(self, latent_dims, embedding_dims, generator_dims, text_embedding_dims=768, 
                 patches_embedding_dims=1024, negative_slope = 0.0, is_bn=False):
        super(generator, self).__init__()

        self.latent_dims = latent_dims
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.embedding_dims = embedding_dims
        self.generator_dims = generator_dims
        self.text_embedding_dims = text_embedding_dims
        self.patches_embedding_dims = patches_embedding_dims
        self.is_bn = is_bn
        self.negative_slope = negative_slope
        self.film_generator = nn.Linear(self.text_embedding_dims, self.patches_embedding_dims * 2)
        self.text_encoder = nn.Linear(self.text_embedding_dims, self.embedding_dims)
        self.patches_encoder = nn.Linear(self.patches_embedding_dims, self.embedding_dims)
        self.patches_transformer_layer = nn.TransformerEncoderLayer(
            d_model=self.embedding_dims, nhead=4, dim_feedforward=self.embedding_dims*2, 
            dropout=0.1, activation='relu', batch_first=True)
        self.patches_cls_token = nn.Parameter(torch.empty(1, 1, self.embedding_dims))
        torch.nn.init.trunc_normal_(self.patches_cls_token, std=0.02)
        self.patches_transformer = nn.TransformerEncoder(self.patches_transformer_layer, num_layers=2)
        self.patch2text_attention = MultiheadAttention(
            embed_dim=self.embedding_dims, num_heads=4, batch_first=True)
        self.text2patch_attention = MultiheadAttention(
            embed_dim=self.embedding_dims, num_heads=4, batch_first=True)
        self.input_dims = self.latent_dims + self.embedding_dims
        self.generator = build_generator(self.input_dims, self.generator_dims[:-1], negative_slope=self.negative_slope, is_bn=is_bn)
        self.final_layer = nn.Linear(self.generator_dims[-2], self.generator_dims[-1])
        
    def forward(self, gene_expression, patches, patches_padding_mask, text_tokens, text_padding_mask):
        gamma_beta = self.film_generator(text_tokens[:, 0, :])
        gamma, beta = gamma_beta.chunk(2, dim=-1)

        gamma = torch.tanh(gamma)
        beta = torch.clamp(beta, min=-5.0, max=5.0)

        gamma = gamma.unsqueeze(1)
        beta = beta.unsqueeze(1)
        patches = gamma * patches + beta

        text_tokens = self.text_encoder(text_tokens)
        patches = self.patches_encoder(patches)

        patches_with_cls = torch.cat((self.patches_cls_token.expand(patches.size(0), -1, -1), patches), dim=1)
        patches_padding_mask = torch.cat((patches_padding_mask.new_zeros(patches_with_cls.size(0), 1, dtype=torch.bool), patches_padding_mask), dim=1)
        patches_with_cls = self.patches_transformer(patches_with_cls, src_key_padding_mask=patches_padding_mask)

        text_cls_vector = text_tokens[:, 0, :self.embedding_dims]
        patch_cls_vector = patches_with_cls[:, 0, :self.embedding_dims]

        patches_with_cls, _ = self.patch2text_attention(
            text_tokens[:, 0:1, :], patches_with_cls, patches_with_cls, key_padding_mask=patches_padding_mask)
        text_tokens, _ = self.text2patch_attention(
            patches_with_cls[:, 0:1, :], text_tokens, text_tokens, key_padding_mask=text_padding_mask)
        text_cls_vector = text_tokens[:, 0, :self.embedding_dims]  # CLS token
        patch_cls_vector = patches_with_cls[:, 0, :self.embedding_dims]  # CLS token
        conditioning_vector = text_cls_vector + patch_cls_vector
        
        x_encoded = torch.cat((gene_expression, conditioning_vector), dim=1)

        for module in self.generator:
            x_encoded = module(x_encoded)

        x_encoded = self.final_layer(x_encoded)

        return x_encoded #, text_embedding, patches


class discriminator(nn.Module):

    def __init__(self, vector_dims, embedding_dims, discriminator_dims, text_embedding_dims=768, 
                 patches_embedding_dims=1024, negative_slope = 0.0, is_bn=False):
        super(discriminator, self).__init__()

        self.vector_dims = vector_dims
        self.embedding_dims = embedding_dims
        self.discriminator_dims = discriminator_dims
        self.text_embedding_dims = text_embedding_dims
        self.patches_embedding_dims = patches_embedding_dims
        self.is_bn = is_bn
        self.negative_slope = negative_slope
        self.film_generator = nn.Linear(self.text_embedding_dims, self.patches_embedding_dims * 2)
        self.text_encoder = nn.Linear(self.text_embedding_dims, self.embedding_dims)
        self.patches_encoder = nn.Linear(self.patches_embedding_dims, self.embedding_dims)
        self.patches_transformer_layer = nn.TransformerEncoderLayer(
            d_model=self.embedding_dims, nhead=4, dim_feedforward=self.embedding_dims*2, 
            dropout=0.1, activation='relu', batch_first=True)
        self.patches_cls_token = nn.Parameter(torch.empty(1, 1, self.embedding_dims))
        torch.nn.init.trunc_normal_(self.patches_cls_token, std=0.02)
        self.patches_transformer = nn.TransformerEncoder(self.patches_transformer_layer, num_layers=2)
        self.patch2text_attention = MultiheadAttention(
            embed_dim=self.embedding_dims, num_heads=4, batch_first=True)
        self.text2patch_attention = MultiheadAttention(
            embed_dim=self.embedding_dims, num_heads=4, batch_first=True)
        self.input_dims = self.vector_dims + self.embedding_dims
        self.discriminator = build_discriminator(self.input_dims, self.discriminator_dims[:-1], negative_slope = self.negative_slope, is_bn=is_bn)
        self.final_layer = nn.Linear(self.discriminator_dims[-2], self.discriminator_dims[-1])

    def forward(self, gene_expression, patches, patches_padding_mask, text_tokens, text_padding_mask): 
        gamma_beta = self.film_generator(text_tokens[:, 0, :])
        gamma, beta = gamma_beta.chunk(2, dim=-1)

        gamma = torch.tanh(gamma)
        beta = torch.clamp(beta, min=-5.0, max=5.0)

        gamma = gamma.unsqueeze(1)
        beta = beta.unsqueeze(1)
        patches = gamma * patches + beta

        text_tokens = self.text_encoder(text_tokens)
        patches = self.patches_encoder(patches)

        patches_with_cls = torch.cat((self.patches_cls_token.expand(patches.size(0), -1, -1), patches), dim=1)
        patches_padding_mask = torch.cat((patches_padding_mask.new_zeros(patches_with_cls.size(0), 1, dtype=torch.bool), patches_padding_mask), dim=1)
        patches_with_cls = self.patches_transformer(patches_with_cls, src_key_padding_mask=patches_padding_mask)

        text_cls_vector = text_tokens[:, 0, :self.embedding_dims]
        patch_cls_vector = patches_with_cls[:, 0, :self.embedding_dims]

        patches_with_cls, _ = self.patch2text_attention(
            text_tokens[:, 0:1, :], patches_with_cls, patches_with_cls, key_padding_mask=patches_padding_mask)
        text_tokens, _ = self.text2patch_attention(
            patches_with_cls[:, 0:1, :], text_tokens, text_tokens, key_padding_mask=text_padding_mask)
        text_cls_vector = text_tokens[:, 0, :self.embedding_dims]  # CLS token
        patch_cls_vector = patches_with_cls[:, 0, :self.embedding_dims]  # CLS token
        conditioning_vector = text_cls_vector + patch_cls_vector
        
        x_encoded = torch.cat((gene_expression, conditioning_vector), dim=1)

        for module in self.discriminator:
            x_encoded = module(x_encoded)

        x_encoded = self.final_layer(x_encoded)

        return x_encoded #, text_embedding, patches


def WGAN_GP_model(latent_dims,
                vector_dims,
                embedding_dims,
                generator_dims,
                discriminator_dims,
                text_embedding_dims=768,
                patches_embedding_dims=1024,
                negative_slope = 0.0, is_bn= False):

    gen  = generator(
        latent_dims, embedding_dims, generator_dims, text_embedding_dims, 
        patches_embedding_dims, negative_slope, is_bn)

    disc = discriminator(
        vector_dims, embedding_dims, discriminator_dims, text_embedding_dims, 
        patches_embedding_dims, negative_slope, is_bn)

    return gen, disc


class WGAN_GP():

    def __init__(self, input_dims, latent_dims,
                embedding_dims,
                generator_dims,
                discriminator_dims,
                text_embedding_dims=768,
                patches_embedding_dims=1024,
                negative_slope = 0.0, is_bn= False,
                lr_d = 5e-4, lr_g = 5e-4, optimizer='rms_prop',
                gp_weight = 10,
                p_aug=0, norm_scale=0.5, train=True,
                n_critic = 5,
                freq_print= 2, freq_compute_test = 50, freq_visualize_test=100, patience=10,
                normalization = 'standardize', log2 = False, rpm = False,
                results_dire = ''):

        self.input_dims = input_dims
        self.latent_dims = latent_dims
        self.embedding_dims = embedding_dims
        self.generator_dims = generator_dims
        self.discriminator_dims = discriminator_dims
        self.text_embedding_dims = text_embedding_dims
        self.patches_embedding_dims = patches_embedding_dims
        self.negative_slope = negative_slope
        self.is_bn = is_bn
        self.gp_weight = gp_weight
        self.isTrain  = train
        self.p_aug = p_aug
        self.norm_scale = norm_scale
        self.n_genes = input_dims
        self.n_critic = n_critic
        self.freq_print= freq_print
        self.freq_compute_test = freq_compute_test
        self.freq_visualize_test = freq_visualize_test
        self.result_dire = results_dire
        os.makedirs(self.result_dire, exist_ok=True)
        self.results_dire_fig = os.path.join(self.result_dire, 'figures')
        os.makedirs(self.results_dire_fig, exist_ok=True)
        self.dend = False
        self.lr_d = lr_d
        self.lr_g = lr_g
        self.optimizer = optimizer
        self.patience = patience
        # Enabling GPU
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print('device:', self.device)
        
        self.loss_dict = {'d loss': [],
                          'd real loss': [],
                          'd fake loss': [],
                          'g loss': []}
        self.corr_scores = {}
        self.corr_dend_scores = {}
        self.precision_scores = {}
        self.recall_scores = {}

        self.normalization = normalization
        self.log2 = log2
        self.rpm = rpm
        print('Normalization:', self.normalization)
        print('Log2:', self.log2)
        print('RPM:', self.rpm)

    def init_train(self):

        # Optimizers
        if self.optimizer.lower() == 'rms_prop':
            self.optimizer_disc = torch.optim.RMSprop(self.disc.parameters(), lr=self.lr_d)
            self.optimizer_gen = torch.optim.RMSprop(self.gen.parameters(), lr=self.lr_g)
        elif self.optimizer.lower() == 'adam':
            self.optimizer_disc = torch.optim.Adam(self.disc.parameters(), lr=self.lr_d, betas=(.9, .99))
            self.optimizer_gen = torch.optim.Adam(self.gen.parameters(), lr=self.lr_g, betas=(.9, .99))
        elif self.optimizer.lower() == 'adamw':
            self.optimizer_disc = torch.optim.AdamW(self.disc.parameters(), lr=self.lr_d, betas=(.9, .99), weight_decay=0.01)
            self.optimizer_gen = torch.optim.AdamW(self.gen.parameters(), lr=self.lr_g, betas=(.9, .99), weight_decay=0.01)

    # Train the discriminator.
    def build_WGAN_GP(self):

        self.numerical_dims = []
        self.gen, self.disc = WGAN_GP_model(
            self.latent_dims, 
            self.input_dims,
            self.embedding_dims,
            self.generator_dims,
            self.discriminator_dims,
            self.text_embedding_dims,
            self.patches_embedding_dims,
            self.negative_slope, 
            self.is_bn)

        self.disc = self.disc.to(self.device)
        self.gen = self.gen.to(self.device)

    def gradient_penalty(self, real_data, fake_data, patches, padding_mask, text_token, text_token_padding):

        batch_size = real_data.size(0)
        alpha= torch.rand(batch_size, 1,
            requires_grad=True,
            device=real_data.device)
        #interpolation =  real_data + torch.mul(alpha, real_data-fake_data)
        interpolation = torch.mul(alpha, real_data) + torch.mul((1 - alpha), fake_data)
        disc_inter_inputs = interpolation
        disc_inter_outputs = self.disc(disc_inter_inputs, patches, padding_mask, text_token, text_token_padding)
        grad_outputs = torch.ones_like(disc_inter_outputs)

        # Compute Gradients
        gradients = torch.autograd.grad(
            outputs=disc_inter_outputs,
            inputs=interpolation,
            grad_outputs=grad_outputs,
            create_graph=True,
            retain_graph=True,)[0]

        # Compute and return Gradient Norm
        gradients = gradients.view(batch_size, -1)
        grad_norm = gradients.norm(2, 1)
        return torch.mean((grad_norm - 1) ** 2)

    def train_disc(self, real_data, z, text_token, text_token_padding, patches, padding_mask):

        self.disc.train()
        batch_size = z.shape[0]
        # clear existing gradient
        self.optimizer_disc.zero_grad()

        # weights update discriminator
        for w in self.disc.parameters():
            w.requires_grad = True

        # no weights update generator
        for w in self.gen.parameters():
            w.requires_grad = False

        gen_outputs = self.gen(z, patches, padding_mask, text_token, text_token_padding)

        # augumentation for stability
        # add noise to both fake and true samples
        if self.p_aug != 0:
            augs = torch.distributions.binomial.Binomial(total_count=1,probs=torch.tensor([self.p_aug])).sample(torch.tensor([batch_size])).to(gen_outputs.device)
            #print(torch.normal(0, self.norm_scale, size=(self.n_genes,)))
            gen_outputs = gen_outputs + augs * torch.normal(0, self.norm_scale, size=(self.n_genes,)).to(gen_outputs.device)
            #gen_outputs = gen_outputs + augs[:, None] * torch.normal(0, self.norm_scale, size=(self.n_genes,)).to(gen_outputs.device)
            #noise = torch.normal(0, self.norm_scale, size=(batch_size, self.n_genes)).to(x.device)
            x = x + augs * torch.normal(0, self.norm_scale, size=(self.n_genes,)).to(x.device)

        disc_fake = self.disc(gen_outputs, patches, padding_mask, text_token, text_token_padding)
        disc_true = self.disc(real_data, patches, padding_mask, text_token, text_token_padding)

        # compute loss
        disc_loss, disc_real_loss, disc_fake_loss = D_loss(disc_true, disc_fake)
        gp = self.gradient_penalty(real_data, gen_outputs, patches, padding_mask, text_token, text_token_padding)
        self.disc_loss = disc_loss + self.gp_weight * gp
        # backprop
        self.disc_loss.requires_grad_(True)
        self.disc_loss.backward()
        # update
        torch.nn.utils.clip_grad_norm_(self.disc.parameters(), max_norm=10.0)
        self.optimizer_disc.step()

        # save batch loss
        '''self.d_batch_loss = np.array([disc_loss.cpu().detach().numpy().tolist(),
                             disc_real_loss.cpu().detach().numpy().tolist(),
                             disc_fake_loss.cpu().detach().numpy().tolist()])'''
        self.d_batch_loss = np.array([disc_loss.item(),
                            disc_real_loss.item(),
                            disc_fake_loss.item()])

    def train_gen(self, z, text_token, text_token_padding, patches, padding_mask):

        self.gen.train()
        batch_size = z.shape[0]
        # clear existing gradient
        self.optimizer_gen.zero_grad()

        # no weights update discriminator
        for w in self.disc.parameters():
            w.requires_grad = False

        # weights update discriminator
        for w in self.gen.parameters():
            w.requires_grad = True

        gen_inputs = z
        gen_outputs = self.gen(gen_inputs, patches, padding_mask, text_token, text_token_padding)
        if self.p_aug != 0:
            augs = torch.distributions.binomial.Binomial(total_count=1,probs=torch.tensor([self.p_aug])).sample(torch.tensor([batch_size])).to(gen_outputs.device)
            gen_outputs = gen_outputs + augs * torch.normal(0, self.norm_scale, size=(self.n_genes,)).to(gen_outputs.device)
            #noise = torch.normal(0, self.norm_scale, size=(batch_size, self.n_genes)).to(gen_outputs.device)
            #gen_outputs = gen_outputs + augs * noise


        disc_fake = self.disc(gen_outputs, patches, padding_mask, text_token, text_token_padding)

        # compute loss
        self.gen_loss = G_loss(disc_fake)
        # backprop
        self.gen_loss.requires_grad_(True)
        self.gen_loss.backward()
        # update
        torch.nn.utils.clip_grad_norm_(self.gen.parameters(), max_norm=2.0)
        self.optimizer_gen.step()

        #self.g_batch_loss =np.array([self.gen_loss.cpu().detach().numpy().tolist()])
        self.g_batch_loss =np.array([self.gen_loss.item()])

    def train(self, gene_expression, text_token, text_token_padding, patches, padding_mask):

        x_real = gene_expression.to(torch.float32).to(self.device)
        text_token = text_token.to(self.device)
        text_token_padding = text_token_padding.to(self.device)
        patches = patches.to(self.device)
        padding_mask = padding_mask.to(self.device)

        # Train critic
        for _ in range(self.n_critic):
            z =  torch.normal(0,1, size=(x_real.shape[0], self.latent_dims), device=self.device)
            self.train_disc(x_real, z, text_token, text_token_padding, patches, padding_mask)
        #Train generator
        z =  torch.normal(0,1, size=(x_real.shape[0], self.latent_dims), device=self.device)
        self.train_gen(z, text_token, text_token_padding, patches, padding_mask)

    def generate_samples_all(self, data_loader, num_repeats=1, balanced=False, balanced_max_oversample=5):
        
        if balanced:
            all_real = []
            all_disease_type_real = []
            all_primary_site_array = []
            for i, batch in enumerate(data_loader):
                gene_expression = batch[2].clone().to(torch.float32)
                all_real.append(gene_expression.numpy())
                all_disease_type_real.append(batch[5].numpy())
                all_primary_site_array.append(batch[6].numpy())
            all_real_x = np.vstack(all_real)
            all_disease_type_real = np.concatenate(all_disease_type_real, axis=0)
            all_primary_site_array = np.concatenate(all_primary_site_array, axis=0)
                
            unique_disease_types = np.unique(all_disease_type_real)
            counts = np.bincount(all_disease_type_real)
            max_count = np.max(counts)
            indices_by_disease = {disease_type: np.where(all_disease_type_real == disease_type)[0] for disease_type in unique_disease_types}

            unique_primary_sites = np.unique(all_primary_site_array)
            primary_site_counts = np.bincount(all_primary_site_array)
            max_primary_site_count = np.max(primary_site_counts)
            indices_by_primary_site = {primary_site: np.where(all_primary_site_array == primary_site)[0] for primary_site in unique_primary_sites}
            
            dataset = data_loader.dataset
            all_gen = []
            all_disease_type_gen = []
            all_primary_site_gen = []
            
            for i in tqdm(range(num_repeats), desc=f"Generating samples ({num_repeats} repeats)", total=num_repeats, unit="repeat"):
                for disease_type in unique_disease_types:
                    disease_indices = indices_by_disease[disease_type]
                    if counts[disease_type] < max_count:
                        num_to_add = max_count - counts[disease_type]
                        if num_to_add > balanced_max_oversample * counts[disease_type]:
                            num_to_add = balanced_max_oversample * counts[disease_type]
                        additional_indices = np.random.choice(disease_indices, num_to_add, replace=num_to_add > len(disease_indices))
                        disease_indices = np.concatenate((disease_indices, additional_indices), axis=0)
                    
                    for start_idx in range(0, len(disease_indices), 64):
                        end_idx = min(start_idx + 64, len(disease_indices))
                        indices = disease_indices[start_idx:end_idx]
                        
                        text_embedding = []
                        gene_expression = []
                        patches = []
                        padding_mask = []
                        disease_type = []
                        primary_site = []
                        for idx in indices:
                            text_embedding.append(dataset[idx][0].unsqueeze(0).to(self.device))
                            text_padding.append(dataset[idx][1].unsqueeze(0).to(self.device))
                            gene_expression.append(dataset[idx][2].unsqueeze(0).to(self.device))
                            patches.append(dataset[idx][3].unsqueeze(0).to(self.device))
                            padding_mask.append(dataset[idx][4].unsqueeze(0).to(self.device))
                            disease_type.append(dataset[idx][5].item())
                            primary_site.append(dataset[idx][6].item())
                            
                        text_embedding = torch.cat(text_embedding, dim=0).to(self.device)
                        text_padding = torch.cat(text_padding, dim=0).to(self.device)
                        gene_expression = torch.cat(gene_expression, dim=0).to(self.device)
                        patches = torch.cat(patches, dim=0).to(self.device)
                        padding_mask = torch.cat(padding_mask, dim=0).to(self.device)
                        disease_type = torch.tensor(disease_type, dtype=torch.long).to(self.device)
                        primary_site = torch.tensor(primary_site, dtype=torch.long).to(self.device)
                        
                        _, x_gen = self.generate_samples(gene_expression, text_embedding, text_padding, patches, padding_mask)
                        all_gen.append(x_gen.cpu().detach().numpy())
                        all_disease_type_gen.append(disease_type.cpu().detach().numpy())
                        all_primary_site_gen.append(primary_site.cpu().detach().numpy())
                        
            all_gen_x = np.vstack(all_gen)
            all_disease_type_gen = np.concatenate(all_disease_type_gen, axis=0)
            all_primary_site_gen = np.concatenate(all_primary_site_gen, axis=0)
            
            indices = np.arange(all_gen_x.shape[0])
            np.random.shuffle(indices)
            all_gen_x = all_gen_x[indices]
            all_disease_type_gen = all_disease_type_gen[indices]
            all_primary_site_gen = all_primary_site_gen[indices]
                
        else:
            
            all_real = []
            all_gen = []
            all_disease_type_real = []
            all_disease_type_gen = []
            all_primary_site_real = []
            all_primary_site_gen = []
            
            for i in tqdm(range(num_repeats), desc=f"Generating samples ({num_repeats} repeats)", total=num_repeats, unit="repeat"):
                for batch in data_loader:
                    text_embedding = batch[0].to(self.device)
                    text_padding = batch[1].to(self.device)
                    gene_expression = batch[2].to(self.device)
                    patches = batch[3].to(self.device)
                    padding_mask = batch[4].to(self.device)
                    disease_type = batch[5].to(self.device)
                    primary_site = batch[6].to(self.device)
                    
                    x_real, x_gen = self.generate_samples(gene_expression, text_embedding, text_padding, patches, padding_mask)

                    all_gen.append(x_gen.cpu().detach().numpy())
                    all_disease_type_gen.append(disease_type.cpu().detach().numpy())
                    all_primary_site_gen.append(primary_site.cpu().detach().numpy())
                    
                    if i == 0:
                        # mi vergogno di questo if che controlla la variabile di iterazione del for ma vabbè pazienza
                        all_real.append(x_real.cpu().detach().numpy())
                        all_disease_type_real.append(disease_type.cpu().detach().numpy())
                        all_primary_site_real.append(primary_site.cpu().detach().numpy())

            all_real_x = np.vstack(all_real)
            all_gen_x = np.vstack(all_gen)
            all_disease_type_gen = np.concatenate(all_disease_type_gen, axis=0)
            all_disease_type_real = np.concatenate(all_disease_type_real, axis=0)
            all_primary_site_gen = np.concatenate(all_primary_site_gen, axis=0)
            all_primary_site_real = np.concatenate(all_primary_site_real, axis=0)

        return all_real_x, all_gen_x, all_disease_type_real, all_disease_type_gen, all_primary_site_real, all_primary_site_gen

    def generate_samples(self, gene_expression, text_embedding, text_padding, patches, padding_mask):
        with torch.no_grad():
            self.gen.eval()
            x_real = gene_expression.clone().to(torch.float32)
            z = torch.normal(0, 1, size=(x_real.shape[0], self.latent_dims), device=self.device)
            x_gen = self.gen(z, patches, padding_mask, text_embedding, text_padding)

        return x_real, x_gen

    def set_requires_grad(self, nets, requires_grad=False):

        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def fit(self, train_data, val_data, test_data, epochs, val=True):
        torch.cuda.init()
        self.build_WGAN_GP()
        total_parameters= sum(p.numel() for p in self.gen.parameters() if p.requires_grad)
        total_non_trainable_parameters= sum(p.numel() for p in self.gen.parameters() if not p.requires_grad)
        print('Total parameters: ', total_parameters)
        print('Total non trainable parameters: ', total_non_trainable_parameters)

        if self.isTrain:
            self.init_train()

        metric_history = {
            'precision': [],
            'recall': [],
            'corr': [],
                     
            'det_lr': [],
            'det_rf': [],
            'det_mlp': [],
            }   
        
        acc_lr_validation= []
        f1_lr_validation = []
        auc_lr_validation= []
        precision_val = []
        recall_val = []

       # early_stopper = EarlyStopper_gan(patience=self.patience, min_delta=0)
        for epoch in range(epochs):

            if epoch % 100 == 0 and epoch != 0:
                print('reducing learning rate')
                for param_group in self.optimizer_disc.param_groups:
                    param_group['lr'] = param_group['lr']*0.50
                    print(f"new lr_d: {param_group['lr']}")

                for param_group in self.optimizer_gen.param_groups:
                    param_group['lr'] = param_group['lr']*0.50
                    print(f"new lr_g: {param_group['lr']}")

            train_loss = 0.0
            print('Epoch: ', epoch)
            self.epoch = epoch
            print('----------Training----------')
            d_loss_all = 0

            for i, data in enumerate(train_data):

                text_tokens = data[0].to(self.device)
                text_tokens_padding = data[1].to(self.device)
                gene_expression = data[2].to(self.device)
                patches = data[3].to(self.device)
                padding_mask = data[4].to(self.device)
     
                self.train(gene_expression, text_tokens, text_tokens_padding, patches, padding_mask)
                d_loss_all += self.disc_loss.item()

                if i==0:
                    d_batch_loss  = self.d_batch_loss
                    g_batch_loss = self.g_batch_loss

                else:
                    d_batch_loss = d_batch_loss + (self.d_batch_loss)
                    g_batch_loss = g_batch_loss + self.g_batch_loss

                if (i+1) % self.freq_print == 0:

                    print('[Epoch %d/%d] [Batch %d/%d] [D loss : %f] [G loss : %f]'
                            %(epoch+1, epochs,       # [Epoch -]
                            i+1,len(train_data),   # [Batch -]
                            self.disc_loss.item(),       # [D loss -]
                            self.gen_loss.item(),       # [G loss -]
                            #loss_GAN.item(),     # [adv -]
                            #loss_cycle.item(),   # [cycle -]
                            ))

            d_batch_loss = d_batch_loss/len(train_data)
            self.loss_dict['d loss'].append(d_batch_loss[0])
            self.loss_dict['d real loss'].append(d_batch_loss[1])
            self.loss_dict['d fake loss'].append(d_batch_loss[2])
            self.loss_dict['g loss'].append(g_batch_loss[0])

            print('Averge D Loss:', d_loss_all/len(train_data))        

            if val:       
                
                if (epoch+1) % self.freq_compute_test == 0:

                    data_real, data_gen, training_disease_type_real, training_disease_type_gen, training_primary_site_real, training_primary_site_gen, = self.generate_samples_all(train_data)
                    all_real, all_gen, testing_disease_type_real, testing_disease_type_gen,testing_primary_site_real, testing_primary_site_gen =  self.generate_samples_all(val_data)

                    torch.save(self.gen.state_dict(), os.path.join(self.result_dire,f'generator_epoch_{epoch+1}.pt'))
                    torch.save(self.disc.state_dict(), os.path.join(self.result_dire,f'discriminator_epoch_{epoch+1}.pt'))
         
                    print("testing_disease_type_real", testing_disease_type_real)
                    
                    plot_umaps(all_real, all_gen, self.results_dire_fig, epoch+1, testing_disease_type_real,  n_neighbors=300)
                    # print('Computing utility (TSTR).......')
                    # results = tissues_classification(all_gen, testing_disease_type_gen ,data_real, training_disease_type_real)
                    # print('Computing utility (TRTR).......')
                    # results = tissues_classification(data_real, all_tissues_training, all_real, all_tissue)
                    
                    
                    results_detection = detection(data_real, data_gen, all_real, all_gen)
                    
                    
                    for model_name in results_detection:
                        if model_name == 'Logistic Regression':
                            acc_lr_validation.append(results_detection[model_name]['accuracy'][0])
                            f1_lr_validation.append(results_detection[model_name]['f1'][0])
                            auc_lr_validation.append(results_detection[model_name]['auc'][0])
                            

                    metrics = compute_evaluation_metrics(data_real, data_gen, all_real, all_gen)
                    precision_val.append(metrics['precision_test'])
                    recall_val.append(metrics['recall_test'])

                    all_results = {}
                    all_detection = {}
                    
                    if (epoch+1) == epochs:
                        print('plot umap....')
                        #plot_umaps(all_real, all_gen, self.results_dire_fig, epoch+1, all_tissue,  n_neighbors=300)

                        torch.save(self.gen.state_dict(), os.path.join(self.result_dire,'generator_last_epoch.pt'))
                        torch.save(self.disc.state_dict(), os.path.join(self.result_dire,'discriminator_last_epoch.pt'))
                        print("Models saved at last epoch.")

                        print("Plot of the training metrics")


                        epochs = [i * self.freq_compute_test for i in range(1, len(acc_lr_validation)+1)]

                        plt.figure(figsize=(12, 6))

                        plt.plot(epochs, acc_lr_validation, label='Accuracy (LR)', marker='o')
                        plt.plot(epochs, f1_lr_validation, label='F1 Score (LR)', marker='s')
                        plt.plot(epochs, auc_lr_validation, label='AUC (LR)', marker='^')
                        plt.plot(epochs, precision_val, label='Precision', linestyle='--', marker='d')
                        plt.plot(epochs, recall_val, label='Recall', linestyle='--', marker='x')

                        plt.xlabel('Epoch')
                        plt.ylabel('Metric Value')
                        plt.title('Validation Metrics Over Epochs')
                        plt.legend()
                        plt.grid(True)
                        plt.tight_layout()

                        #plt.savefig("validation_metrics.png")
                        plt.savefig(os.path.join(self.result_dire, 'validation_metrics.png'))
                        plt.close()

                        n_runs = 2
                        precision = []
                        recall = []
                        corr = []
                        f1_lr = []
                        f1_mlp = []
                        f1_rf = []
                        acc_lr = []
                        
                        acc_mlp = []
                        acc_rf = []
                        auc_lr = []
                        auc_mlp = []
                        auc_rf = []
                        for run in range(n_runs):
                            print('run:', run)
                            print('----------Testing----------')
                            data_real, data_gen, all_tissue_train, all_tissue_train_gen, all_primary_sites_training, gen_primary_sites = self.generate_samples_all(train_data)
                            
                            all_real, all_gen, all_tissue_test, all_tissue_test_gen, all_primary_sites, all_gen_primary_sites =  self.generate_samples_all(test_data)
                        
                            # must save the data for final utility evaluation    
                            results_dire_run = os.path.join(self.result_dire, f"test_{run}_epoch_{epoch+1}")
                            create_folder(results_dire_run)
                            save_numpy(results_dire_run + '/data_real.npy', data_real)
                            save_numpy(results_dire_run + '/data_gen.npy', data_gen)
                            save_numpy(results_dire_run + '/test_real.npy', all_real)
                            save_numpy(results_dire_run + '/test_gen.npy', all_gen)                            
                            save_numpy(results_dire_run + '/train_labels_real.npy', all_tissue_train)
                            save_numpy(results_dire_run + '/train_labels_gen.npy', all_tissue_train_gen)
                            save_numpy(results_dire_run + '/test_labels_real.npy', all_tissue_test)
                            save_numpy(results_dire_run + '/test_labels_gen.npy', all_tissue_test_gen)                            
                            save_numpy(results_dire_run + '/train_primary_site_real.npy', all_primary_sites_training)
                            save_numpy(results_dire_run + '/train_primary_site_gen.npy', gen_primary_sites)
                            save_numpy(results_dire_run + '/test_primary_site_real.npy', all_primary_sites)
                            save_numpy(results_dire_run + '/test_primary_site_gen.npy', all_gen_primary_sites)

                            corr.append(gamma_coef(all_real, all_gen))
                            metrics = compute_evaluation_metrics(data_real, data_gen, all_real, all_gen)
                            precision.append(metrics['precision_test'])
                            recall.append(metrics['recall_test'])
                            all_results[str(run)] = metrics
                            print(metrics)
                            #wandb.log({"accuracy detection": metrics['Logistic results'][1], "f1 detection":  metrics['Logistic results'][0]})
                            #wandb.log({"accuracy detection pca": metrics['Logistic PCA results'][1], "f1 detection pca":  metrics['Logistic PCA results'][0]})
                            print('-------------------------------------------------------------------------------------------')
                            print(f"Detection complete feature space with {data_real.shape[1]} features")
                            results_detection = detection(data_real, data_gen, all_real, all_gen)
                            all_detection[str(run)] = results_detection
                            acc = []
                            f1 = []
                            auc = []
                            for model_name in results_detection:
                        
                                acc.append(results_detection[model_name]['accuracy'][0])
                                f1.append(results_detection[model_name]['f1'][0])
                                auc.append(results_detection[model_name]['auc'][0])

                                if model_name == 'Logistic Regression':
                                    f1_lr.append(results_detection[model_name]['f1'][0])
                                    acc_lr.append(results_detection[model_name]['accuracy'][0])
                                    auc_lr.append(results_detection[model_name]['auc'][0])
                                    
                                elif model_name == 'Random Forest':
                                    f1_rf.append(results_detection[model_name]['f1'][0])
                                    acc_rf.append(results_detection[model_name]['accuracy'][0])
                                    auc_rf.append(results_detection[model_name]['auc'][0])
                                else:
                                    f1_mlp.append(results_detection[model_name]['f1'][0])
                                    acc_mlp.append(results_detection[model_name]['accuracy'][0])
                                    auc_mlp.append(results_detection[model_name]['auc'][0])
                                       
                            print(f"Model: {model_name}, Accuracy: {results_detection[model_name]['accuracy']}, F1: {results_detection[model_name]['f1']}, 'AUC': {results_detection[model_name]['auc']}")
                            print('-------------------------------------------------------------------------------------------')
                            
                            n_components = 100
                            pca = PCA(n_components=n_components)
                            pca_train_data = pca.fit_transform(data_real)
                            pca_gen_data = pca.transform(data_gen)
                            pca_data_real_test = pca.transform(all_real)
                            pca_data_fake_test = pca.transform(all_gen)
                            print(f"Detection PCA space with {pca_data_real_test.shape[1]} PCs")
                            results_detection = detection(pca_train_data, pca_gen_data, 
                                             pca_data_real_test, pca_data_fake_test)

                            all_detection[str(run) + '_PCA'] = results_detection
                            acc = []
                            f1 = []
                            auc = []
                            for model_name in results_detection:
                        
                                acc.append(results_detection[model_name]['accuracy'][0])
                                f1.append(results_detection[model_name]['f1'][0])
                                auc.append(results_detection[model_name]['auc'][0])
                        
                            print(f"Model: {model_name}, Accuracy: {results_detection[model_name]['accuracy']}, F1: {results_detection[model_name]['f1']}, 'AUC': {results_detection[model_name]['auc']}")   
                            print('-------------------------------------------------------------------------------------------')                                                
                            print('Training completed!')    
                                
                        def mean_std(values):
                            return np.mean(values), np.std(values)
                        
                        precision_mean, precision_std = mean_std(precision)
                        recall_mean, recall_std = mean_std(recall)
                        corr_mean, corr_std = mean_std(corr)
                        f1_lr_mean, f1_lr_std = mean_std(f1_lr)
                        f1_mlp_mean, f1_mlp_std = mean_std(f1_mlp)
                        f1_rf_mean, f1_rf_std = mean_std(f1_rf)
                        acc_lr_mean, acc_lr_std = mean_std(acc_lr)
                        acc_mlp_mean, acc_mlp_std = mean_std(acc_mlp)
                        acc_rf_mean, acc_rf_std = mean_std(acc_rf)
                        auc_lr_mean, auc_lr_std = mean_std(auc_lr)
                        auc_mlp_mean, auc_mlp_std = mean_std(auc_mlp)
                        auc_rf_mean, auc_rf_std = mean_std(auc_rf)
                       
                        # Stampa formattata con media ± deviazione standard
                        print(f"Precisione: {precision_mean:.4f} ± {precision_std:.4f}")
                        print(f"Recall: {recall_mean:.4f} ± {recall_std:.4f}")
                        print(f"Correlazione: {corr_mean:.4f} ± {corr_std:.4f}")
                        print(f"F1-score - LR: {f1_lr_mean:.4f} ± {f1_lr_std:.4f}, MLP: {f1_mlp_mean:.4f} ± {f1_mlp_std:.4f}, RF: {f1_rf_mean:.4f} ± {f1_rf_std:.4f}")
                        print(f"Accuratezza - LR: {acc_lr_mean:.4f} ± {acc_lr_std:.4f}, MLP: {acc_mlp_mean:.4f} ± {acc_mlp_std:.4f}, RF: {acc_rf_mean:.4f} ± {acc_rf_std:.4f}")
                        print(f"AUC - LR: {auc_lr_mean:.4f} ± {auc_lr_std:.4f}, MLP: {auc_mlp_mean:.4f} ± {auc_mlp_std:.4f}, RF: {auc_rf_mean:.4f} ± {auc_rf_std:.4f}")

        return self.loss_dict

    def print_best_epoch(self, d, name='correlation'):
        idx_max = max(d, key=d.get)
        print('Best epoch ' + name + ':', idx_max, 'score:', d[idx_max])


    
def parse_args():
    parser = argparse.ArgumentParser(description="Train a conditional GAN model")
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--num_epochs', type=int, default=500, help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training')
    parser.add_argument('--latent_dim', type=int, default=256, help='Latent dimension for the model')
    parser.add_argument('--hidden_dim', type=int, default=256, help='Hidden dimension for the model')
    parser.add_argument('--embedding_dim', type=int, default=256, help='Embedding dimension for the model')
    parser.add_argument('--num_patches', type=int, default=256, help='Number of patches for multi-patch model')
    parser.add_argument('--dataset_path', type=str, default='/CompanyDatasets/carlos00/all_tcga_100mb', help='Path to the dataset')
    parser.add_argument('--output_path', type=str, default='/Experiments/carlos00/iciap/conditional_gan', help='Path to save the model')
    parser.add_argument('--num_workers', type=int, default=16, help='Number of workers for data loading')
    parser.add_argument('--freq_compute_test', type=int, default= 50, help='Frequency of validation performance')
    parser.add_argument('--freq_plot_images', type=int, default=16, help='Frequency for the plot')
    parser.add_argument('--optimizer', type=str, default='rms_prop', help='Optimizer to use for training')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    
    print(f'Arguments: {args.__dict__}')
    
    train_loader, validation_loader, test_loader, n_genes = dataloader_multi_patch_conditional_gan(
        Path(args.dataset_path),
        normalize=True,
        percentage_to_remove=90,
        norm_type='standardize',
        num_patches=args.num_patches,
        batch_size=args.batch_size,
        seed=args.seed,
        num_workers=args.num_workers,
        embedding_dim=args.embedding_dim,
        text_embedding_file='clinical_modernbert_embeddings.parquet',
        patch_embeddings_folder='patch_embeddings_uni',
        token_embeddings_folder='../clinical_modernbert_embeddings')

    h = args.hidden_dim
    model = WGAN_GP(
        input_dims= n_genes, 
        latent_dims=args.latent_dim,
        embedding_dims=args.embedding_dim,   
        text_embedding_dims=768,
        patches_embedding_dims=1024,
        generator_dims=[h, h, n_genes],
        discriminator_dims=[h, h, 1],
        optimizer=args.optimizer,
        negative_slope=0.0, is_bn=False,
        lr_d=5e-4, lr_g=5e-4, gp_weight=10,
        p_aug=0, norm_scale=0.5, freq_compute_test=args.freq_compute_test, results_dire=args.output_path)                

    d_l = model.fit(train_loader, validation_loader, test_loader, epochs=args.num_epochs)

    print("--------- Disease Type Evaluation ----------")

    evaluator = UtilityEvaluator(results_path=args.output_path)
    evaluator.evaluate()
    evaluator.report()
    
    print("--------- Primary Site Evaluation ----------")
    
    evaluator = UtilityEvaluatorPrimary(results_path=args.output_path)
    evaluator.evaluate()
    evaluator.report()

    print()
    print("--------- Privacy Evaluation ----------")

    def load_data(folder):
        return {
            "data_real": np.load(os.path.join(folder, 'data_real.npy')),
            "data_gen": np.load(os.path.join(folder, 'data_gen.npy')),
            "test_real": np.load(os.path.join(folder, 'test_real.npy')),
            "test_gen": np.load(os.path.join(folder, 'test_gen.npy')),
        }

    results_dirs = sorted(glob(os.path.join(args.output_path, 'test_*')))
    dcr_scores = []
    nndr_scores = []
    for results_dir in results_dirs:
        data = load_data(results_dir)
        dcr_score = dcr(data['data_real'], data['data_gen'], data['test_real'])
        nndr_score = nndr(data['data_real'], data['data_gen'], data['test_real'])
        dcr_scores.append(dcr_score)
        nndr_scores.append(nndr_score)
    
    mean_dcr = np.mean(dcr_scores)
    std_dcr = np.std(dcr_scores)
    mean_nndr = np.mean(nndr_scores)
    std_nndr = np.std(nndr_scores)
    print(f"DCR {mean_dcr:.4f}±{std_dcr:.4f}, NNDR {mean_nndr:.4f}±{std_nndr:.4f}")