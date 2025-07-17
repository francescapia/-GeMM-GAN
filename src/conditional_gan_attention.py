import torch
from torch import nn, Tensor
import torch.nn.functional as F
import numpy as np
from torch.nn import MultiheadAttention
from sklearn.decomposition import PCA 
import os
from generative_model_utils import *
from unsupervised_metrics import *
from corr_score import * 
from multi_patch_gan_dataloader import *
import warnings
warnings.filterwarnings("ignore")
from tqdm import tqdm
from pathlib import Path
import argparse
from datetime import datetime
from utility_evaluation import UtilityEvaluator
from utility_primary_s_evaluation import UtilityEvaluatorPrimary


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
            generator.append(build_linear_block(generator_dims[i-1], generator_dims[i], negative_slope= negative_slope, is_bn= is_bn))
    return generator

def build_discriminator(input_dims, dicriminator_dims, negative_slope= 0.0, is_bn=False):

    dicriminator = nn.ModuleList()
    for i in range(len(dicriminator_dims)):
        if i == 0:
            dicriminator.append(build_linear_block(input_dims, dicriminator_dims[i], negative_slope= negative_slope, is_bn=is_bn))
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
        self.text_encoder = nn.Linear(self.text_embedding_dims, self.embedding_dims)
        self.patches_encoder = nn.Linear(self.patches_embedding_dims, self.embedding_dims)
        self.attention = MultiheadAttention(embed_dim=self.embedding_dims, num_heads=4, batch_first=True)
        self.input_dims = self.latent_dims + self.embedding_dims
        self.attn_bn = nn.BatchNorm1d(embedding_dims)
        self.generator = build_generator(self.input_dims, self.generator_dims[:-1], negative_slope=self.negative_slope, is_bn=is_bn)
        self.final_layer = nn.Linear(self.generator_dims[-2], self.generator_dims[-1])
        
    def forward(self, x, text_embedding, patches, padding_mask):
        text_embedding = self.text_encoder(text_embedding)
        patches = self.patches_encoder(patches)

        query = text_embedding.unsqueeze(1)
        attn_output, _ = self.attention(query, patches, patches, key_padding_mask=padding_mask)
        attn_vector = attn_output.squeeze(1)
        
        print('batch norm')
        print("before", attn_vector)
        attn_vector = self.attn_bn(attn_vector)
        print("after", attn_vector)
        x_encoded = torch.cat((x,attn_vector), dim=1)

        for module in self.generator:
            x_encoded = module(x_encoded)
            
        x_encoded  = self.final_layer(x_encoded)
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
        self.text_encoder = nn.Linear(self.text_embedding_dims, self.embedding_dims)
        self.patches_encoder = nn.Linear(self.patches_embedding_dims, self.embedding_dims)
        self.attention = MultiheadAttention(embed_dim=self.embedding_dims, num_heads=4, batch_first=True)
        self.input_dims = self.vector_dims + self.embedding_dims
        self.discriminator = build_discriminator(self.input_dims, self.discriminator_dims[:-1], negative_slope = self.negative_slope, is_bn=is_bn)
        self.final_layer = nn.Linear(self.discriminator_dims[-2], self.discriminator_dims[-1])

    def forward(self, gene_expression, text_embedding, patches, padding_mask): 
        text_embedding = self.text_encoder(text_embedding)
        patches = self.patches_encoder(patches)

        query = text_embedding.unsqueeze(1)
        attn_output, _ = self.attention(query, patches, patches, key_padding_mask=padding_mask)
        attn_vector = attn_output.squeeze(1)
        
        x_encoded = torch.cat((gene_expression, attn_vector), dim=1)

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

    def gradient_penalty(self, real_data, fake_data, text_embedding, patches, padding_mask):

        batch_size = real_data.size(0)
        alpha= torch.rand(batch_size, 1,
            requires_grad=True,
            device=real_data.device)
        #interpolation =  real_data + torch.mul(alpha, real_data-fake_data)
        interpolation = torch.mul(alpha, real_data) + torch.mul((1 - alpha), fake_data)
        disc_inter_inputs = interpolation
        disc_inter_outputs = self.disc(disc_inter_inputs, text_embedding, patches, padding_mask)
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

    def train_disc(self, real_data, z, text_embedding, patches, padding_mask):

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

        gen_outputs = self.gen(z, text_embedding, patches, padding_mask)

        # augumentation for stability
        # add noise to both fake and true samples
        if self.p_aug != 0:
            augs = torch.distributions.binomial.Binomial(total_count=1,probs=torch.tensor([self.p_aug])).sample(torch.tensor([batch_size])).to(gen_outputs.device)
            #print(torch.normal(0, self.norm_scale, size=(self.n_genes,)))
            gen_outputs = gen_outputs + augs * torch.normal(0, self.norm_scale, size=(self.n_genes,)).to(gen_outputs.device)
            #gen_outputs = gen_outputs + augs[:, None] * torch.normal(0, self.norm_scale, size=(self.n_genes,)).to(gen_outputs.device)
            #noise = torch.normal(0, self.norm_scale, size=(batch_size, self.n_genes)).to(x.device)
            x = x + augs * torch.normal(0, self.norm_scale, size=(self.n_genes,)).to(x.device)

        disc_fake = self.disc(gen_outputs, text_embedding, patches, padding_mask)
        disc_true = self.disc(real_data, text_embedding, patches, padding_mask)

        # compute loss
        disc_loss, disc_real_loss, disc_fake_loss = D_loss(disc_true, disc_fake)
        gp = self.gradient_penalty(real_data, gen_outputs, text_embedding, patches, padding_mask)
        self.disc_loss = disc_loss + self.gp_weight * gp
        # backprop
        self.disc_loss.requires_grad_(True)
        self.disc_loss.backward()
        # update
        self.optimizer_disc.step()

        # save batch loss
        '''self.d_batch_loss = np.array([disc_loss.cpu().detach().numpy().tolist(),
                             disc_real_loss.cpu().detach().numpy().tolist(),
                             disc_fake_loss.cpu().detach().numpy().tolist()])'''
        self.d_batch_loss = np.array([disc_loss.item(),
                            disc_real_loss.item(),
                            disc_fake_loss.item()])

    def train_gen(self, z, text_embedding, patches, padding_mask):

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
        gen_outputs = self.gen(gen_inputs, text_embedding, patches, padding_mask)
        if self.p_aug != 0:
            augs = torch.distributions.binomial.Binomial(total_count=1,probs=torch.tensor([self.p_aug])).sample(torch.tensor([batch_size])).to(gen_outputs.device)
            gen_outputs = gen_outputs + augs * torch.normal(0, self.norm_scale, size=(self.n_genes,)).to(gen_outputs.device)
            #noise = torch.normal(0, self.norm_scale, size=(batch_size, self.n_genes)).to(gen_outputs.device)
            #gen_outputs = gen_outputs + augs * noise


        disc_fake = self.disc(gen_outputs, text_embedding, patches, padding_mask)

        # compute loss
        self.gen_loss = G_loss(disc_fake)
        # backprop
        self.gen_loss.requires_grad_(True)
        self.gen_loss.backward()
        # update
        self.optimizer_gen.step()

        #self.g_batch_loss =np.array([self.gen_loss.cpu().detach().numpy().tolist()])
        self.g_batch_loss =np.array([self.gen_loss.item()])

    def train(self, gene_expression, text_embedding, patches, padding_mask):

        x_real = gene_expression.to(torch.float32).to(self.device)
        text_embedding = text_embedding.to(self.device)
        patches = patches.to(self.device)
        padding_mask = padding_mask.to(self.device)

        # Train critic
        for _ in range(self.n_critic):
            z =  torch.normal(0,1, size=(x_real.shape[0], self.latent_dims), device=self.device)
            self.train_disc(x_real, z, text_embedding, patches, padding_mask)
        #Train generator
        z =  torch.normal(0,1, size=(x_real.shape[0], self.latent_dims), device=self.device)
        self.train_gen(z, text_embedding, patches, padding_mask)

    def generate_samples_all(self, data_loader, num_repeats=1, balanced=False, balanced_max_oversample=5):
        
        if balanced:
            all_real = []
            all_disease_type_real = []
            for i, batch in enumerate(data_loader):
                gene_expression = batch[1].clone().to(torch.float32)
                all_real.append(gene_expression.numpy())
                all_disease_type_real.append(batch[4].numpy())
            all_real_x = np.vstack(all_real)
            all_disease_type_real = np.concatenate(all_disease_type_real, axis=0)
                
            unique_disease_types = np.unique(all_disease_type_real)
            counts = np.bincount(all_disease_type_real)
            max_count = np.max(counts)
            indices_by_disease = {disease_type: np.where(all_disease_type_real == disease_type)[0] for disease_type in unique_disease_types}
            
            dataset = data_loader.dataset
            all_gen = []
            all_disease_type_gen = []
            
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
                        for idx in indices:
                            text_embedding.append(dataset[idx][0].unsqueeze(0).to(self.device))
                            gene_expression.append(dataset[idx][1].unsqueeze(0).to(self.device))
                            patches.append(dataset[idx][2].unsqueeze(0).to(self.device))
                            padding_mask.append(dataset[idx][3].unsqueeze(0).to(self.device))
                            disease_type.append(dataset[idx][4].item())
                            
                        text_embedding = torch.cat(text_embedding, dim=0).to(self.device)
                        gene_expression = torch.cat(gene_expression, dim=0).to(self.device)
                        patches = torch.cat(patches, dim=0).to(self.device)
                        padding_mask = torch.cat(padding_mask, dim=0).to(self.device)
                        disease_type = torch.tensor(disease_type, dtype=torch.long).to(self.device)
                        
                        _, x_gen = self.generate_samples(gene_expression, text_embedding, patches, padding_mask)
                        all_gen.append(x_gen.cpu().detach().numpy())
                        all_disease_type_gen.append(disease_type.cpu().detach().numpy())
                        
            all_gen_x = np.vstack(all_gen)
            all_disease_type_gen = np.concatenate(all_disease_type_gen, axis=0)
            
            indices = np.arange(all_gen_x.shape[0])
            np.random.shuffle(indices)
            all_gen_x = all_gen_x[indices]
            all_disease_type_gen = all_disease_type_gen[indices]
                
        else:
            
            all_real = []
            all_gen = []
            all_disease_type_real = []
            all_disease_type_gen = []
            
            for i in tqdm(range(num_repeats), desc=f"Generating samples ({num_repeats} repeats)", total=num_repeats, unit="repeat"):
                for batch in data_loader:
                    text_embedding = batch[0].to(self.device)
                    gene_expression = batch[1].to(self.device)
                    patches = batch[2].to(self.device)
                    padding_mask = batch[3].to(self.device)
                    disease_type = batch[4].to(self.device)
                    
                    x_real, x_gen = self.generate_samples(gene_expression, text_embedding, patches, padding_mask)

                    all_gen.append(x_gen.cpu().detach().numpy())
                    all_disease_type_gen.append(disease_type.cpu().detach().numpy())
                    
                    if i == 0:
                        # mi vergogno di questo if che controlla la variabile di iterazione del for ma vabbè pazienza
                        all_real.append(x_real.cpu().detach().numpy())
                        all_disease_type_real.append(disease_type.cpu().detach().numpy())

            all_real_x = np.vstack(all_real)
            all_gen_x = np.vstack(all_gen)
            all_disease_type_gen = np.concatenate(all_disease_type_gen, axis=0)
            all_disease_type_real = np.concatenate(all_disease_type_real, axis=0)


        return all_real_x, all_gen_x, all_disease_type_real, all_disease_type_gen

    def generate_samples(self, gene_expression, text_embedding, patches, padding_mask):
        with torch.no_grad():
            self.gen.eval()
            x_real = gene_expression.clone().to(torch.float32)
            z = torch.normal(0, 1, size=(x_real.shape[0], self.latent_dims), device=self.device)
            x_gen = self.gen(z, text_embedding, patches, padding_mask)

        return x_real, x_gen

    def set_requires_grad(self, nets, requires_grad=False):

        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def fit(self, train_data, test_data, epochs, val=True):
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

       # early_stopper = EarlyStopper_gan(patience=self.patience, min_delta=0)
        for epoch in range(epochs):

            if epoch % 50 == 0 and epoch != 0:
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

                text_embedding = data[0].to(self.device)
                gene_expression = data[1].to(self.device)
                patches = data[2].to(self.device)
                padding_mask = data[3].to(self.device)
     
                self.train(gene_expression, text_embedding, patches, padding_mask)
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

                    all_results = {}
                    all_detection = {}
                    
                    if (epoch+1) == epochs:
                        print('plot umap....')
                        #plot_umaps(all_real, all_gen, self.results_dire_fig, epoch+1, all_tissue,  n_neighbors=300)

                        torch.save(self.gen.state_dict(), os.path.join(self.result_dire,'generator_last_epoch.pt'))
                        torch.save(self.disc.state_dict(), os.path.join(self.result_dire,'discriminator_last_epoch.pt'))
                        print("Models saved at last epoch.")

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
                            data_real, data_gen, training_disease_type_real, training_disease_type_gen = self.generate_samples_all(train_data)
                            
                            all_real, all_gen, testing_disease_type_real, testing_disease_type_gen =  self.generate_samples_all(test_data)
                        
                            # must save the data for final utility evaluation    
                            results_dire_run = os.path.join(self.result_dire, f"test_{run}_epoch_{epoch+1}")
                            create_folder(results_dire_run)
                            save_numpy(results_dire_run + '/data_real.npy', data_real)
                            save_numpy(results_dire_run + '/data_gen.npy', data_gen)
                            save_numpy(results_dire_run + '/test_real.npy', all_real)
                            save_numpy(results_dire_run + '/test_gen.npy', all_gen)                            
                            save_numpy(results_dire_run + '/train_labels_real.npy', training_disease_type_real)
                            save_numpy(results_dire_run + '/train_labels_gen.npy', training_disease_type_gen)
                            save_numpy(results_dire_run + '/test_labels_real.npy', testing_disease_type_real)
                            save_numpy(results_dire_run + '/test_labels_gen.npy', testing_disease_type_gen)                            

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
    
    def evaluate(self, train_data, val_data, test_data, epochs):
        
        torch.cuda.init()
        self.build_WGAN_GP()
        
        self.gen.load_state_dict(torch.load(self.result_dire + '/generator_last_epoch.pt'))
        self.disc.load_state_dict(torch.load(self.result_dire + '/discriminator_last_epoch.pt'))
        self.gen.to('cuda:0').eval()
        self.disc.to('cuda:0').eval()
        
        all_results = {}
        all_detection = {}
        
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
            data_real, data_gen, training_disease_type_real, training_disease_type_gen = self.generate_samples_all(train_data)
            
            all_real, all_gen, testing_disease_type_real, testing_disease_type_gen =  self.generate_samples_all(test_data)
        

            # must save the data for final utility evaluation
            
            results_dire_run = os.path.join(self.result_dire, f"test_{run}_epoch_500")
            create_folder(results_dire_run)


            save_numpy(results_dire_run + '/data_real.npy', data_real)
            save_numpy(results_dire_run + '/data_gen.npy', data_gen)
            save_numpy(results_dire_run + '/test_real.npy', all_real)
            save_numpy(results_dire_run + '/test_gen.npy', all_gen)                            
            save_numpy(results_dire_run + '/train_labels_real.npy', training_disease_type_real)
            save_numpy(results_dire_run + '/train_labels_gen.npy', training_disease_type_gen)
            save_numpy(results_dire_run + '/test_labels_real.npy', testing_disease_type_real)
            save_numpy(results_dire_run + '/test_labels_gen.npy', testing_disease_type_gen)
        

            
            
            

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
        num_workers=args.num_workers)

    h = args.hidden_dim
    model = WGAN_GP(
        input_dims= n_genes, 
        latent_dims=args.latent_dim,
        embedding_dims=args.embedding_dim,   
        generator_dims=[h, h, n_genes],
        discriminator_dims=[h, h, 1],
        negative_slope=0.0, is_bn=False,
        lr_d=5e-4, lr_g=5e-4, gp_weight=10,
        p_aug=0, norm_scale=0.5, freq_compute_test = args.freq_compute_test, results_dire=args.output_path)                

    d_l = model.fit(train_loader, validation_loader, test_loader, epochs=args.num_epochs)
    
    evaluator = UtilityEvaluator(results_path=args.output_path)
    evaluator.evaluate()
    evaluator.report()
    
    print("--------- Primary Site Evaluation ----------")
    
    evaluator = UtilityEvaluatorPrimary(results_path=args.output_path)
    evaluator.evaluate()
    evaluator.report()