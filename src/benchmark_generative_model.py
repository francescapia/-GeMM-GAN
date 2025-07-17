import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import argparse
import pickle    
import os
from glob import glob
import shutil
from pathlib import Path
from sklearn.decomposition import PCA 
import os
from generative_model_utils import *
from unsupervised_metrics import *
from corr_score import * 
import warnings
warnings.filterwarnings("ignore")
from torch_geometric.nn import GraphConv, global_mean_pool
from utility_evaluation import UtilityEvaluator
from utility_primary_s_evaluation import UtilityEvaluatorPrimary
from benchmark_gan_dataloader import dataloader_benchmark_conditional_gan


def categorical_embedding(vocab_sizes):

    #n_cat_vars = len(vocab_sizes)
    embedder = nn.ModuleList()
    for vs in vocab_sizes:
        emdedding_dims = 128 #int(vs**0.5) +1
        embedder.append(nn.Embedding(vs,  emdedding_dims))
    
    return embedder

def save_numpy(file, data):
    with open(file, 'wb') as f:
        np.save(f, data)

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

def build_discriminator(input_dims, dicriminator_dims, negative_slope= 0.0, is_bn=False):

    dicriminator = nn.ModuleList()
    for i in range(len(dicriminator_dims)):
        if i == 0:
            dicriminator.append(build_linear_block(input_dims, dicriminator_dims[i], negative_slope= negative_slope, is_bn=is_bn))
        else:
            dicriminator.append(build_linear_block(dicriminator_dims[i-1], dicriminator_dims[i], negative_slope=negative_slope, is_bn=is_bn))
    return dicriminator

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

def build_generator(input_dims, generator_dims, negative_slope = 0.0, is_bn = False):

    generator = nn.ModuleList()
    for i in range(len(generator_dims)):
        if i == 0:
            print(input_dims)
            generator.append(build_linear_block(input_dims, generator_dims[i], negative_slope=negative_slope, is_bn= is_bn))
        else:
            generator.append(build_linear_block(generator_dims[i-1], generator_dims[i], negative_slope= negative_slope, is_bn= is_bn))
    return generator


class discriminator(nn.Module):

    def __init__(self, vector_dims,
                numerical_dims,
                vocab_sizes,
               discriminator_dims, negative_slope = 0.0, is_bn=False):
        super(discriminator, self).__init__()
        '''
        Take as input a gene expression sample and try to distinguish the true inputs
        '''

        self.vector_dims = vector_dims
        self.numerical_dims = len(numerical_dims)
        print('numerical_dims:', self.numerical_dims)
        self.vocab_sizes = vocab_sizes
        self.discriminator_dims = discriminator_dims
        self.negative_slope = negative_slope
        self.n_cat_vars = len(self.vocab_sizes)
        self.categorical_embedded_dims = 256 #sum([(int(vs**0.5)+1) for vs in self.vocab_sizes])
        self.input_dims = self.vector_dims + self.numerical_dims +  self.categorical_embedded_dims
        print(self.input_dims)

        self.discriminator = build_discriminator(self.input_dims, self.discriminator_dims[:-1], negative_slope = self.negative_slope, is_bn=is_bn)
        self.final_layer = nn.Linear(self.discriminator_dims[-2], self.discriminator_dims[-1])
        # categorical embeddings
        self.categorical_embedding = categorical_embedding(self.vocab_sizes)
        #self.embedding = nn.Embedding( self.vocab_size,  self.categorical_embedded_dims)


    def forward(self, x, categorical_covariates,categorical_covariates_2):

        embedded_cat_vars = []
        #cat_x = self.embeddinokg(categorical_covariates)
        for i, module in enumerate(self.categorical_embedding):
            #cat_x = categorical_covariates[:,i]
            if i==0:
                cat_x = categorical_covariates
            else:
                
                cat_x = categorical_covariates_2  
                 
            embedded_cat_vars.append(module(cat_x))
        if self.n_cat_vars == 1:
            embedded_cat_vars = embedded_cat_vars[0]

        else:
            embedded_cat_vars = torch.cat(embedded_cat_vars, dim=1)

        x_encoded = torch.cat((x,embedded_cat_vars), dim=1)

        for module in self.discriminator:
            x_encoded = module(x_encoded)

        x_encoded = self.final_layer(x_encoded)
        
        #print(x_encoded)
        return x_encoded
    
    
class generator(nn.Module):

    def __init__(self, latent_dims,
                numerical_dims,
                vocab_sizes,
               generator_dims,
              negative_slope = 0.0, is_bn=False):
        super(generator, self).__init__()

        '''
        Parameters:

        '''


        self.latent_dims = latent_dims
        self.numerical_dims = len(numerical_dims)
        print('numerical_dims:', self.numerical_dims)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.vocab_sizes = vocab_sizes
        self.generator_dims = generator_dims
        self.negative_slope = negative_slope
        self.n_cat_vars = len(self.vocab_sizes)
        # concatenate noise vector + numerical covariates + embedded categorical covariates (e.g., tissue type)
        #self.input_dims = self.latent_dims+ self.numerical_dims +  self.categorical_embedded_dims * self.vocab_size
        self.categorical_embedded_dims = 256 #sum([(int(vs**0.5)+1)  for vs in self.vocab_sizes])
        self.input_dims = self.latent_dims + self.numerical_dims +  self.categorical_embedded_dims

        self.generator = build_generator(self.input_dims, self.generator_dims[:-1], negative_slope=self.negative_slope, is_bn=is_bn)
        self.final_layer = nn.Linear(self.generator_dims[-2], self.generator_dims[-1])
        self.final_activation = nn.ReLU()
        self.relu = nn.ReLU()
        # categorical embeddings
        self.categorical_embedding = categorical_embedding(self.vocab_sizes)



      
    def forward(self, x, categorical_covariates,categorical_covariates_2 ):

        embedded_cat_vars = []
        #cat_x = self.embedding(categorical_covariates)
        for i, module in enumerate(self.categorical_embedding):
            if i==0:
               
                cat_x = categorical_covariates
            else:
            
                cat_x = categorical_covariates_2
                
            embedded_cat_vars.append(module(cat_x))
            
        if self.n_cat_vars == 1:
            embedded_cat_vars = embedded_cat_vars[0]

        else:
            embedded_cat_vars = torch.cat(embedded_cat_vars,dim=1)


       # print("embedded_cat_vars shape", embedded_cat_vars.shape)

        x = torch.cat((x,embedded_cat_vars), dim=1)
        x_encoded = x


        for module in self.generator:

            x_encoded = module(x_encoded)

        #x_encoded = self.final_activation(self.final_layer(x_encoded))
        x_encoded  = self.final_layer(x_encoded)
        #x_encoded = self.threshold(x_encoded)
        # normalize the output



        return x_encoded

def WGAN_GP_model_benchmark(latent_dims,
                vector_dims,
                numerical_dims,
                vocab_sizes,
                generator_dims,
                discriminator_dims,
                negative_slope = 0.0, is_bn= False):

    gen  = generator(latent_dims,
               numerical_dims,
               vocab_sizes,
               generator_dims,
               negative_slope, is_bn)

    disc = discriminator(vector_dims,
                                numerical_dims,
                                vocab_sizes,
                                discriminator_dims,
                                negative_slope, is_bn)

    return gen, disc





class WGAN_GP_benchmark():


    def __init__(self, input_dims, latent_dims,
                vocab_sizes,
                generator_dims,
                discriminator_dims,
                negative_slope = 0.0, is_bn= False,  numerical_dims= [],
                lr_d = 5e-4, lr_g = 5e-4, optimizer='rms_prop',
                gp_weight = 10,
                p_aug=0, norm_scale=0.5, train=True,
                n_critic = 5,
                freq_print= 2, freq_compute_test = 10, freq_visualize_test=100, patience=10,
                normalization = 'standardize', log2 = False, rpm = False,
                results_dire = ''):

        self.input_dims = input_dims
        self.latent_dims = latent_dims
        self.numerical_dims = numerical_dims,
        self.vocab_sizes = vocab_sizes
        self.generator_dims = generator_dims
        self.discriminator_dims = discriminator_dims
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
        #self.device = torch.device('cpu')
        #self.loss_fn = utils.set_loss(self.opt,self.device)
        print('numerical_dims:', len(numerical_dims))
        self.loss_dict = {'d loss': [],
                          'd real loss': [],
                          'd fake loss': [],
                          'g loss': []}
        self.corr_scores = {}
        self.corr_dend_scores = {}
        self.precision_scores = {}
        self.recall_scores = {}
        #self.stat1 = torch.tensor(stat1, dtype=torch.float32).to(self.device)
        #self.stat2 = torch.tensor(stat2, dtype=torch.float32).to(self.device)


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

        # to do: fix bug
        self.numerical_dims = []
        self.gen, self.disc = WGAN_GP_model_benchmark(self.latent_dims, self.input_dims, self.numerical_dims,
                self.vocab_sizes,
                self.generator_dims,
                self.discriminator_dims,
                self.negative_slope, self.is_bn)

        self.disc = self.disc.to(self.device)
        self.gen = self.gen.to(self.device)

    def gradient_penalty(self, real_data, fake_data, cat_vars,cat_vars2):

        batch_size = real_data.size(0)
        alpha= torch.rand(batch_size, 1,
            requires_grad=True,
            device=real_data.device)
        #interpolation =  real_data + torch.mul(alpha, real_data-fake_data)
        interpolation = torch.mul(alpha, real_data) + torch.mul((1 - alpha), fake_data)
        disc_inter_inputs = interpolation
        disc_inter_outputs = self.disc(disc_inter_inputs,  cat_vars,cat_vars2)
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


    def train_disc(self, x, z, cat_vars, cat_vars2):


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

              # generator input -> concatenate z + numerical vars


        gen_inputs = z
        gen_outputs = self.gen(gen_inputs, cat_vars,cat_vars2)


        # augumentation for stability
        # add noise to both fake and true samples
        if self.p_aug != 0:
            augs = torch.distributions.binomial.Binomial(total_count=1,probs=torch.tensor([self.p_aug])).sample(torch.tensor([batch_size])).to(gen_outputs.device)
            #print(torch.normal(0, self.norm_scale, size=(self.n_genes,)))
            gen_outputs = gen_outputs + augs * torch.normal(0, self.norm_scale, size=(self.n_genes,)).to(gen_outputs.device)
            #gen_outputs = gen_outputs + augs[:, None] * torch.normal(0, self.norm_scale, size=(self.n_genes,)).to(gen_outputs.device)
            #noise = torch.normal(0, self.norm_scale, size=(batch_size, self.n_genes)).to(x.device)
            x = x + augs * torch.normal(0, self.norm_scale, size=(self.n_genes,)).to(x.device)

        disc_fake = self.disc(gen_outputs, cat_vars,cat_vars2)
        disc_true = self.disc(x, cat_vars,cat_vars2)

        # compute loss
        disc_loss, disc_real_loss, disc_fake_loss = D_loss(disc_true, disc_fake)
        gp = self.gradient_penalty(x, gen_outputs, cat_vars,cat_vars2)
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

    def train_gen(self, z, cat_vars,cat_vars_2):


        self.gen.train()
        batch_size = z.shape[0]
        # clear existing gradient
        self.optimizer_gen.zero_grad()


         # no weights update discriminator
        for w in self.disc.parameters():
            w.requires_grad = False

        #  weights update discriminator
        for w in self.gen.parameters():
            w.requires_grad = True

        gen_inputs = z
        gen_outputs = self.gen(gen_inputs, cat_vars,cat_vars_2)
        if self.p_aug != 0:
            augs = torch.distributions.binomial.Binomial(total_count=1,probs=torch.tensor([self.p_aug])).sample(torch.tensor([batch_size])).to(gen_outputs.device)
            gen_outputs = gen_outputs + augs * torch.normal(0, self.norm_scale, size=(self.n_genes,)).to(gen_outputs.device)
            #noise = torch.normal(0, self.norm_scale, size=(batch_size, self.n_genes)).to(gen_outputs.device)
            #gen_outputs = gen_outputs + augs * noise


        disc_fake = self.disc(gen_outputs, cat_vars, cat_vars_2)

        # compute loss
        self.gen_loss = G_loss(disc_fake)
        # backprop
        self.gen_loss.requires_grad_(True)
        self.gen_loss.backward()
        # update
        self.optimizer_gen.step()

        #self.g_batch_loss =np.array([self.gen_loss.cpu().detach().numpy().tolist()])
        self.g_batch_loss =np.array([self.gen_loss.item()])

    def train(self, x_GE, x_cat,x_cat_2):


        x_real = x_GE.clone().to(torch.float32)
        # Train critic
        for _ in range(self.n_critic):
            z =  torch.normal(0,1, size=(x_real.shape[0], self.latent_dims), device=self.device)
            self.train_disc(x_real, z, x_cat,x_cat_2)
        #Train generator
        z =  torch.normal(0,1, size=(x_real.shape[0], self.latent_dims), device=self.device)
        self.train_gen(z, x_cat,x_cat_2)

    def generate_samples_all(self, data):

        all_real  = []
        all_gen = []

        all_tissue = []
        balanced_x_cat_array=[]
        primary_site_array = []
        for i, data in enumerate(data):

            x_GE = data[0].to(self.device)
            x_cat = data[1].to(self.device)
            x_cat_2 = data[2].to(self.device)
            tissue_t= data[1].t()
            primary_site= data[2].t()
            num_elements = len(x_cat)


            x_real, x_gen = self.generate_samples(x_GE, x_cat,x_cat_2)
            if i==0:
                all_gen =  x_gen.cpu().detach().numpy()
                all_real = x_real.cpu().detach().numpy()
            else:
                all_gen = np.append(all_gen,  x_gen.cpu().detach().numpy(), axis=0)
                all_real = np.append(all_real, x_real.cpu().detach().numpy(), axis=0)

            balanced_x_cat_array.extend([x.cpu().numpy() for x in x_cat])
            primary_site_array.extend([x.cpu().numpy() for x in primary_site])
        #all_tissue,_ = remap_labels(torch.tensor(all_tissue))


        all_real_x = np.vstack(all_real)
        all_real_gen = np.vstack(all_gen)

        #print("all_tissue_label", all_tissue)

        return all_real_x, all_real_gen, balanced_x_cat_array,balanced_x_cat_array,primary_site_array,primary_site_array#all_tissue


    def generate_samples(self, x_GE,  x_cat,x_cat_2):

        with torch.no_grad():
            self.gen.eval()
            x_real = x_GE.clone().to(torch.float32)
            #z =  torch.normal(0,1, size=(x_real.shape[0], self.latent_dims), device=self.device)
            z =  torch.normal(0,1, size=(x_cat.shape[0], self.latent_dims), device=self.device)
            gen_inputs = z
            x_gen = self.gen(gen_inputs, x_cat, x_cat_2)

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
                       
            'ut_bacc_lr': [],
            'ut_bacc_rf': [],
            'ut_bacc_mlp': [],
                    
            'ut_bf1_lr': [],
            'ut_bf1_rf': [],
            'ut_bf1_mlp': [],
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

                x_GE = data[0].to(self.device)
                x_cat = data[1].to(self.device)
                x_cat_2 = data[2].to(self.device)
                #print("x_cat shape", x_cat)
                #print("x_cat shape", x_cat_2)
                #print(torch.cuda.memory_summary())
                self.train(x_GE,x_cat,x_cat_2)
                d_loss_all += self.disc_loss.item()

                tissue_t= x_cat.t()

                if i==0:
                    d_batch_loss  = self.d_batch_loss
                    g_batch_loss = self.g_batch_loss

                else:
                    d_batch_loss = d_batch_loss + (self.d_batch_loss)
                    g_batch_loss = g_batch_loss + self.g_batch_loss
            # print loss

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
                    all_utility_TSTR ={}
                    all_utility_TRTR ={}
                    all_utility_TRTS ={}
                    
                    if (epoch+1) == epochs:
                        print('plot umap....')
                        #plot_umaps(all_real, all_gen, self.results_dire_fig, epoch+1, all_tissue,  n_neighbors=300)
                        
                        torch.save(self.gen.state_dict(), os.path.join(self.result_dire,'generator_last_epoch.pt'))
                        torch.save(self.disc.state_dict(), os.path.join(self.result_dire,'discriminator_last_epoch.pt'))
            
                        
                      
                        n_runs = 10
                        precision = []
                        recall = []
                        corr = []
                        f1_lr = []
                        f1_mlp = []
                        f1_rf = []
                        acc_lr = []
                        ut_bacc_lr = []
                        ut_bacc_rf = []
                        ut_bacc_mlp = []
                        ut_bf1_lr = []
                        ut_bf1_rf = []
                        ut_bf1_mlp = []
                        acc_mlp = []
                        acc_rf = []
                        auc_lr = []
                        auc_mlp = []
                        auc_rf = []
                        for run in range(n_runs):
                            print('run:', run)
                            print('----------Testing----------')
                            data_real, data_gen, gen_tissues, all_tissues_training, gen_primary_sites,all_primary_sites_training = self.generate_samples_all(train_data)
                            
                            all_real, all_gen, all_tissue_test, all_tissue, all_gen_primary_sites,all_primary_sites =  self.generate_samples_all(test_data)
                        

                            print("gen_tissues_test", pd.DataFrame(all_tissue_test).value_counts())
                            print("real_tissues_test", pd.DataFrame(all_tissue).value_counts())
                            # data_real_renorm,data_gen_renorm,all_real_renorm, all_gen_renorm= reverse_normalization(data_real, data_gen, 
                            #                     all_real, all_gen, self.stat1,
                            #                     self.stat2, self.normalization, self.log2,self.rpm)
            
                            
                            results_dire_run = os.path.join(self.result_dire, f"test_{run}_epoch_{epoch+1}")
                            create_folder(results_dire_run)

                            save_numpy(results_dire_run + '/tissue_training.npy', all_tissues_training)
                            save_numpy(results_dire_run + '/data_real.npy', data_real)
                            save_numpy(results_dire_run + '/data_gen.npy', data_gen)
                            save_numpy(results_dire_run + '/test_real.npy', all_real)
                            save_numpy(results_dire_run + '/test_gen.npy', all_gen)                            
                            save_numpy(results_dire_run + '/tissue_test.npy', all_tissue)
                            
                            save_numpy(results_dire_run + '/train_primary_site_real.npy', all_primary_sites_training)
                            save_numpy(results_dire_run + '/train_primary_site_gen.npy', gen_primary_sites)
                            save_numpy(results_dire_run + '/test_primary_site_real.npy', all_primary_sites)
                            save_numpy(results_dire_run + '/test_primary_site_gen.npy', all_gen_primary_sites)
                        
                            dict_data_real = {'data_train': data_real, 'data_test': all_real}
                            dict_data_real = {'data_train': data_gen, 'data_test': all_gen}
                            
                            
                            print("computing last evaluation with renormalized data........")

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
                            
                            
                            
                            
                            # print(f"TSTR complete feature space with {data_real.shape[1]} features")
                            # #results_utility_TSTR = tissues_classification(data_gen_, all_tissues_training ,all_real, all_tissue)
                        
                           
                            # results_utility_TSTR = tissues_classification(all_gen, all_tissue_test ,all_real, all_tissue)

                            # all_utility_TSTR[str(run)] = results_utility_TSTR
                            # acc = []
                            # balanced_acc = []
                            # f1 = []
                            # f1_weighted = []
                            # for model_name in results_utility_TSTR:
                        
                            #     acc.append(results_utility_TSTR[model_name]['accuracy'][0])
                            #     balanced_acc.append(results_utility_TSTR[model_name]['balanced accuracy'][0])
                            #     f1.append(results_utility_TSTR[model_name]['f1'][0])
                            #     f1_weighted.append(results_utility_TSTR[model_name]['f1_weighted'][0])

                            #     if model_name == 'Logistic Regression':
                            #         ut_bf1_lr.append(results_utility_TSTR[model_name]['f1_weighted'][0])
                            #         ut_bacc_lr.append(results_utility_TSTR[model_name]['balanced accuracy'][0])
                                    
                                    
                            #     elif model_name == 'Random Forest':
                            #         ut_bf1_rf.append(results_utility_TSTR[model_name]['f1'][0])
                            #         ut_bacc_rf.append(results_utility_TSTR[model_name]['balanced accuracy'][0])
                                    
                            #     else:
                            #         ut_bf1_mlp.append(results_utility_TSTR[model_name]['f1'][0])
                            #         ut_bacc_mlp.append(results_utility_TSTR[model_name]['balanced accuracy'][0])
                                    
                        
                            
                            # print('-------------------------------------------------------------------------------------------')   


                            # print(f"TRTR complete feature space with {data_real.shape[1]} features")
                            # results_utility_TRTR = tissues_classification(all_real, all_tissue ,data_real, all_tissues_training)
                            # all_utility_TRTR[str(run)] = results_utility_TRTR
                            # acc = []
                            # balanced_acc = []
                            # f1 = []
                            # f1_weighted = []
                            # for model_name in results_utility_TRTR:
                        
                            #     acc.append(results_utility_TRTR[model_name]['accuracy'][0])
                            #     balanced_acc.append(results_utility_TRTR[model_name]['balanced accuracy'][0])
                            #     f1.append(results_utility_TRTR[model_name]['f1'][0])
                            #     f1_weighted.append(results_utility_TRTR[model_name]['f1_weighted'][0])
                        
                            
                            # print('-------------------------------------------------------------------------------------------')   

                            # print('Training completed!')    
                                
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
                        # ut_bacc_lr_mean, ut_bacc_lr_std = mean_std(ut_bacc_lr)
                        # ut_bacc_mlp_mean, ut_bacc_mlp_std = mean_std(ut_bacc_mlp)
                        # ut_bacc_rf_mean, ut_bacc_rf_std = mean_std(ut_bacc_rf)
                        
                        # ut_bf1_lr_mean, ut_bf1_lr_std = mean_std(ut_bf1_lr)
                        # ut_bf1_mlp_mean, ut_bf1_mlp_std = mean_std(ut_bf1_mlp)
                        # ut_bf1_rf_mean, ut_bf1_rf_std = mean_std(ut_bf1_rf)

                        # Stampa formattata con media ± deviazione standard
                        print(f"Precisione: {precision_mean:.4f} ± {precision_std:.4f}")
                        print(f"Recall: {recall_mean:.4f} ± {recall_std:.4f}")
                        print(f"Correlazione: {corr_mean:.4f} ± {corr_std:.4f}")
                        print(f"F1-score - LR: {f1_lr_mean:.4f} ± {f1_lr_std:.4f}, MLP: {f1_mlp_mean:.4f} ± {f1_mlp_std:.4f}, RF: {f1_rf_mean:.4f} ± {f1_rf_std:.4f}")
                        print(f"Accuratezza - LR: {acc_lr_mean:.4f} ± {acc_lr_std:.4f}, MLP: {acc_mlp_mean:.4f} ± {acc_mlp_std:.4f}, RF: {acc_rf_mean:.4f} ± {acc_rf_std:.4f}")
                        print(f"AUC - LR: {auc_lr_mean:.4f} ± {auc_lr_std:.4f}, MLP: {auc_mlp_mean:.4f} ± {auc_mlp_std:.4f}, RF: {auc_rf_mean:.4f} ± {auc_rf_std:.4f}")
                        # print(f"Utility F1-SCORE - LR: {ut_bf1_lr_mean:.4f} ± {ut_bf1_lr_std:.4f}, MLP: {ut_bf1_mlp_mean:.4f} ± {ut_bf1_mlp_std:.4f}, RF: {ut_bf1_rf_mean:.4f} ± {ut_bf1_rf_std:.4f}")
                        # print(f"Utility Accuratezza - LR: {ut_bacc_lr_mean:.4f} ± {ut_bacc_lr_std:.4f}, MLP: {ut_bacc_mlp_mean:.4f} ± {ut_bacc_mlp_std:.4f}, RF: {ut_bacc_rf_mean:.4f} ± {ut_bacc_rf_std:.4f}")


                        # results_path = "/home/pinoli/genai/data/L_1000/results_wgangp/metrics_summary.txt"  # Customize your folder and filename

                        # with open(results_path, "w") as f:
                        # # Create a function to log and print at once
                        #     def log(msg):
                        #         print(msg)
                        #         f.write(msg + "\n")

                        #     log(f"Precisione: {precision_mean:.4f} ± {precision_std:.4f}")
                        #     log(f"Recall: {recall_mean:.4f} ± {recall_std:.4f}")
                        #     log(f"Correlazione: {corr_mean:.4f} ± {corr_std:.4f}")
                        #     log(f"F1-score - LR: {f1_lr_mean:.4f} ± {f1_lr_std:.4f}, MLP: {f1_mlp_mean:.4f} ± {f1_mlp_std:.4f}, RF: {f1_rf_mean:.4f} ± {f1_rf_std:.4f}")
                        #     log(f"Accuratezza - LR: {acc_lr_mean:.4f} ± {acc_lr_std:.4f}, MLP: {acc_mlp_mean:.4f} ± {acc_mlp_std:.4f}, RF: {acc_rf_mean:.4f} ± {acc_rf_std:.4f}")
                        #     log(f"AUC - LR: {auc_lr_mean:.4f} ± {auc_lr_std:.4f}, MLP: {auc_mlp_mean:.4f} ± {auc_mlp_std:.4f}, RF: {auc_rf_mean:.4f} ± {auc_rf_std:.4f}")
                        #     log(f"Utility F1-SCORE - LR: {ut_bf1_lr_mean:.4f} ± {ut_bf1_lr_std:.4f}, MLP: {ut_bf1_mlp_mean:.4f} ± {ut_bf1_mlp_std:.4f}, RF: {ut_bf1_rf_mean:.4f} ± {ut_bf1_rf_std:.4f}")
                        #     log(f"Utility Accuratezza - LR: {ut_bacc_lr_mean:.4f} ± {ut_bacc_lr_std:.4f}, MLP: {ut_bacc_mlp_mean:.4f} ± {ut_bacc_mlp_std:.4f}, RF: {ut_bacc_rf_mean:.4f} ± {ut_bacc_rf_std:.4f}")
    
    def print_best_epoch(self, d, name='correlation'):
        
        idx_max = max(d, key=d.get)
        print('Best epoch ' + name + ':', idx_max, 'score:', d[idx_max])
    
      # Evaluation metrics
    def score_fn(x_test, cat_covs_test, num_covs_test):

        def _score(gen):
            x_gen = predict(cc=cat_covs_test,
                            nc=num_covs_test,
                            gen=gen)

            gamma_dx_dz = gamma_coefficients(x_test, x_gen)
            return gamma_dx_dz
            # score = (x_test - x_gen) ** 2
            # return -np.mean(score)

        return _score
   
   
def parse_args():
    parser = argparse.ArgumentParser(description='WGAN-GP')
    parser.add_argument('--dataset_path', type=str, default='/CompanyDatasets/carlos00/all_tcga_100mb', help='path to dataset')
    parser.add_argument('--output_path', type=str, default='/Experiments/carlos00/iciap/gan/benchmark_gan', help='path to save the model')
    parser.add_argument('--batch_size', type=int, default=8, help='input batch size for training (default: 8)')
    parser.add_argument('--epochs', type=int, default=500, help='number of epochs to train (default: 500)')
    parser.add_argument('--latent_dim', type=int, default=256, help='latent dimensions (default: 256)')
    parser.add_argument('--num_workers', type=int, default=4, help='number of workers for data loading (default: 16)')
    parser.add_argument('--seed', type=int, default=42, help='random seed (default: 42)')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    train_loader, val_loader, test_loader, n_genes = dataloader_benchmark_conditional_gan(
        dataset_path=Path(args.dataset_path),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed)
    
    with open(f'{args.dataset_path}/metainfos.pkl', 'rb') as f:
        metainfos = pickle.load(f)
        
    with open(f'{args.dataset_path}/case_ids.txt', 'r') as f:
        case_ids = f.read().splitlines()
    case_ids = [c.strip() for c in case_ids]
        
    disease_types = [m['disease_type'] for c, m in metainfos.items() if c in case_ids]
    num_disease_types = len(set(disease_types))
    print(set(disease_types))
    
    primary_sites = [m['primary_site'] for c, m in metainfos.items() if c in case_ids]
    num_primary_sites = len(set(primary_sites))
    print(set(primary_sites))
    
    output_path = args.output_path
    model = WGAN_GP_benchmark(
        input_dims=n_genes, 
        latent_dims=args.latent_dim,
        vocab_sizes=[num_disease_types, num_primary_sites],
        generator_dims=[256, 256, n_genes],
        discriminator_dims=[256, 256, 1],
        negative_slope=0.0, 
        is_bn=False,
        lr_d=5e-4, 
        lr_g=5e-4, 
        gp_weight=10,
        p_aug=0, 
        norm_scale=0.5, 
        results_dire=output_path)
            
    d_l = model.fit(train_loader, test_loader, epochs=args.epochs)

    test_dirs = sorted(glob(os.path.join(output_path, 'test_*')))

    for dir in test_dirs:
        training_path = os.path.join(dir, 'tissue_training.npy')
        testing_path = os.path.join(dir, 'tissue_test.npy')
        shutil.copy(training_path, os.path.join(dir, 'train_labels_real.npy'))
        shutil.copy(training_path, os.path.join(dir, 'train_labels_gen.npy'))
        shutil.copy(testing_path, os.path.join(dir, 'test_labels_real.npy'))
        shutil.copy(testing_path, os.path.join(dir, 'test_labels_gen.npy'))
    
    print("--------- Disease Types Evaluation ----------")
    evaluator = UtilityEvaluator(results_path=output_path)
    evaluator.evaluate()
    evaluator.report()
    
    
    print("--------- Primary Site Evaluation ----------")
    
    evaluator = UtilityEvaluatorPrimary(results_path=args.output_path)
    evaluator.evaluate()
    evaluator.report()