import torch
import torch.nn as nn

import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import argparse
import pickle    

from glob import glob
import shutil
from pathlib import Path
from sklearn.decomposition import PCA 

from generative_model_utils import *
from unsupervised_metrics import *
from corr_score import * 
import warnings
warnings.filterwarnings("ignore")
from torch_geometric.nn import GraphConv, global_mean_pool
from utility_evaluation import UtilityEvaluator
from utility_primary_s_evaluation import UtilityEvaluatorPrimary
from benchmark_gan_dataloader import dataloader_benchmark_conditional_gan

from model_utils_vae import *
from losses import *
# set seed
SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)
random.seed(SEED)

def save_numpy(file, data):
    with open(file, 'wb') as f:
        np.save(f, data)

def gaussian_likelihood(self, x_hat, logscale, x):
        scale = torch.exp(logscale)
        mean = x_hat
        dist = torch.distributions.Normal(mean, scale)

        # measure prob of seeing image under p(x|z)
        log_pxz = dist.log_prob(x)
        return log_pxz.sum()

def categorical_embedding(vocab_sizes):

    #n_cat_vars = len(vocab_sizes)
    embedder = nn.ModuleList()
    for vs in vocab_sizes:
        emdedding_dims = 128 #int(vs**0.5) +1
        embedder.append(nn.Embedding(vs,  emdedding_dims))
    
    return embedder

def kl_divergence_old(z, mu, std):

    # Monte Carlo KL divergence
    #  This means we sample z many times and estimate the KL divergence.
    # in practice, these estimates are really good and with a batch size of 128 or more, 
    # the estimate is very accurate

    p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
    q = torch.distributions.Normal(mu, std)

    log_pz = p.log_prob(z)
    log_qzx = q.log_prob(z)

    kl = (log_qzx - log_pz)

    kl = kl.sum(-1)

    return kl

def kl_divergence(mu, std):

    # - KL_loss = 0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp()) 
    kl = torch.mean(-0.5 * torch.sum(1 + std - mu ** 2 - std.exp(), dim = 1), dim = 0)

    return kl

def reconstruction_loss(x, x_pred, logscale, MSE=True):

    if not MSE:

        # heteroscedastic
        scale = torch.exp(logscale)
        mean = x_pred
        dist = torch.distributions.Normal(mean, scale)

        loss = dist.log_prob(x).sum()
        
    else:
        #loss_1 = ((x - x_pred)**2).sum()
        #loss_1 = ((x - x_pred)**2).mean()
        #print('mse 1:', loss_1)
        loss = nn.functional.mse_loss(x_pred, x)
        #print('mse 2:', loss)
    
    return loss



class VAE_model(nn.Module):
    def __init__(self, input_dims, encoder_dims, latent_dims):
        super(VAE_model, self).__init__()
        self.input_dims = input_dims
        self.latent_dims = latent_dims
        self.encoder_dims = encoder_dims
        self.decoder_dims =  encoder_dims[::-1]
        self.encoder_output_dim = encoder_dims[-1]
  
        self.encoder = build_encoder(self.input_dims, self.encoder_dims)
        self.decoder = build_decoder(self.latent_dims, self.decoder_dims, self.input_dims)

        # add class parameters -> 

        # distribution parameters
        self.mu = nn.Linear(self.encoder_output_dim, self.latent_dims)
        self.log_var = nn.Linear(self.encoder_output_dim, self.latent_dims)
        # exponential activation to ensure var is positive 
        # reconstruction loss

        self.log_scale = nn.Parameter(torch.tensor([0.0]))

    def forward(self, x):
        
        x_encoded = x
        for module in self.encoder:
            x_encoded = module(x_encoded)
        mu, log_var = self.mu(x_encoded), self.log_var(x_encoded)
        #torch.exp(0.5 * logvar)
        q = torch.distributions.Normal(0,1)
        z = mu + torch.exp(log_var*0.5)* q.rsample(mu.shape).to(self.device)
        # q = torch.distributions.Normal(mu, torch.exp(log_var))
        # z = q.rsample(mu.shape)

        self.kl = kl_divergence(mu, torch.exp(log_var))

        x_pred = z
        for module in self.decoder:
            x_pred = module(x_pred)
        
        self.reconstruction_loss = reconstruction_loss(x, x_pred, self.log_scale)
        return x_pred

class CVAE_model(nn.Module):
    def __init__(self, input_dims, encoder_dims, latent_dims,  
                 vocab_sizes, 
                negative_slope = 0.0, is_bn=False):
        super(CVAE_model, self).__init__()
        self.input_dims = input_dims
        self.latent_dims = latent_dims
        self.encoder_dims = encoder_dims
        self.decoder_dims =  encoder_dims[::-1]
        self.encoder_output_dim = encoder_dims[-1]
        
        self.negative_slope = negative_slope
        self.vocab_sizes = vocab_sizes
        self.n_cat_vars = len(self.vocab_sizes)
        self.categorical_embedded_dims = 256 #sum([int(vs**0.5)+1 for vs in self.vocab_sizes])
        self.input_dec_dims = self.latent_dims+  self.categorical_embedded_dims 
        self.encoder = build_encoder(self.input_dims, self.encoder_dims, negative_slope=self.negative_slope, is_bn=is_bn)
        self.decoder = build_decoder(self.input_dec_dims, self.decoder_dims, self.input_dims, 
                                    negative_slope=self.negative_slope, is_bn=is_bn)

        # add class parameters -> 

        # distribution parameters
        self.mu = nn.Linear(self.encoder_output_dim, self.latent_dims)
        self.log_var = nn.Linear(self.encoder_output_dim, self.latent_dims)
        # exponential activation to ensure var is positive 
        # reconstruction loss
        self.categorical_embedding = categorical_embedding(self.vocab_sizes)
        self.log_scale = nn.Parameter(torch.tensor([0.0]))
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    def decode(self, z,  categorical_covariates,categorical_covariates_2):

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

        #z = torch.randn(x_num.shape[0], self.latent_dims)
        #x_pred = torch.cat((z, x_num, x_cat), axis=1)
            
        x_pred = torch.cat((z, embedded_cat_vars), axis=1)
        for module in self.decoder:
            x_pred = module(x_pred)
        return x_pred

    def reparameterize(self, mu, log_var):
      
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, x, categorical_covariates,categorical_covariates_2):
        
      

        x_encoded = x
        for module in self.encoder:
            x_encoded = module(x_encoded)

        mu, log_var = self.mu(x_encoded), self.log_var(x_encoded)

        #q = torch.distributions.Normal(0,1)
        #print(q.rsample(mu.shape).get_device())
      
        # reparametrization
        #z = mu + torch.exp(log_var*0.5)* q.rsample(mu.shape).to(self.device)
        z = self.reparameterize(mu, log_var)
        self.kl = kl_divergence(mu, torch.exp(log_var))
    
        embedded_cat_vars = []
        #cat_x = self.embedding(categorical_covariates)
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

        x_pred = torch.cat((z, embedded_cat_vars), axis=1)
        for module in self.decoder:
            x_pred = module(x_pred)
        
        self.reconstruction_loss = reconstruction_loss(x, x_pred, self.log_scale)
        return x_pred


class CVAE():
    
    def __init__(self, input_dims, encoder_dims, latent_dims,  
                vocab_sizes, beta=1.0, 
                negative_slope = 0.0, is_bn=False, lr = 5e-4, train=True,
                freq_print=2,
                freq_compute_test= 2, freq_visualize_test=100, normalization = 'standardize',
                results_dire = ''):

        self.input_dims = input_dims
        self.latent_dims = latent_dims

        self.encoder_dims = encoder_dims
        self.decoder_dims =  encoder_dims[::-1]
        self.encoder_output_dim = encoder_dims[-1]
        self.is_bn = is_bn
        self.negative_slope = negative_slope
        self.vocab_sizes = vocab_sizes
        self.isTrain  = train
        self.dend = False
        self.lr = lr
        self.freq_print= freq_print
        self.freq_compute_test= freq_compute_test
        self.freq_visualize_test = freq_visualize_test
        self.beta = beta 
        self.results_dire = results_dire
        #self.results_dire_fig = os.path.join(self.results_dire, 'figures')
        #create_folder(self.results_dire)
        os.makedirs(self.results_dire, exist_ok=True)
        #create_folder(self.results_dire_fig)

        # Enabling GPU
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.loss_dict = {'kl loss': [], 
                          'rec real loss': [], 
                          'tot loss': []}
        self.corr_scores = {}
        self.corr_dend_scores = {}
        self.precision_scores = {}
        self.recall_scores = {}
        self.normalization = normalization

        print('Normalization:', self.normalization)


        
    def build_CVAE(self):

        self.CVAE_model = CVAE_model(self.input_dims, self.encoder_dims, self.latent_dims,  
                 self.vocab_sizes, self.negative_slope, self.is_bn)

        self.CVAE_model.to(self.device)

    def init_train(
            self,
            optimizer: str = 'adam'):

        # Optimizers
        if optimizer.lower() == 'rms_prop':
            self.optimizer = torch.optim.RMSprop(self.CVAE_model.parameters(), lr=self.lr)
        
        elif optimizer.lower() == 'adam':
           self.optimizer = torch.optim.Adam(self.CVAE_model.parameters(), lr=self.lr)
    
    def train(self,  x_GE, x_cat, x_cat_2):

        x_real = x_GE.clone().to(torch.float32)
        self.CVAE_model.train()
        self.optimizer.zero_grad()

        outputs = self.CVAE_model(x_real, x_cat,x_cat_2)

        loss_elbo = self.beta * self.CVAE_model.kl + self.CVAE_model.reconstruction_loss
    
        self.loss_elbo = loss_elbo
        self.loss_elbo.backward()
        self.optimizer.step()

        self.batch_losses = np.array([self.loss_elbo.item(),
                                    - self.CVAE_model.kl.item(),
                                    self.CVAE_model.reconstruction_loss.item()])

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

        return all_real_x, all_real_gen, balanced_x_cat_array,balanced_x_cat_array,primary_site_array,primary_site_array
    
    def generate_samples(self, x_GE,  x_cat,x_cat_2):
        
        with torch.no_grad():

            self.CVAE_model.eval()
            x_real = x_GE.clone().to(torch.float32)
            z =  torch.normal(0,1, size=(x_GE.shape[0], self.latent_dims), device=self.device)
            #decoder_inputs = torch.cat((z, x_num.to(torch.float32), x_cat), axis=1)
            x_gen = self.CVAE_model.decode(z, x_cat,x_cat_2)
       
        return x_real, x_gen

    def test(self, test_data, return_labels=False, compute_score=True):

        all_real  = []
        all_gen = []

        all_tissue = []
       
        print('----------Testing----------')
        for i, data in enumerate(test_data):
            
            x_GE = data[0].to(self.device)
            x_cat = data[1].to(self.device)
    
            x_real, x_gen = self.generate_samples(x_GE, x_cat)
            if i==0:
                all_gen =  x_gen.cpu().detach().numpy()
                all_real = x_real.cpu().detach().numpy()
            else: 
                all_gen = np.append(all_gen,  x_gen.cpu().detach().numpy(), axis=0)
                all_real = np.append(all_real, x_real.cpu().detach().numpy(), axis=0)
            
            data_t = data[1].t()
            all_tissue.extend(data_t[0].numpy())
            
        if compute_score:
            all_real_x = np.vstack(all_real)
            print(all_real_x.shape)
            all_real_gen = np.vstack(all_gen)
            print(all_real_gen.shape)
            if self.dend:
                pass
                # gamma_dx_dz, gamma_tx_tz = gamma_coefficients(all_real_x, all_real_gen)
                # print(gamma_dx_dz)
                # print(gamma_tx_tz)
                # self.corr_dend_scores[self.epoch +1] = gamma_tx_tz
            else: 
                print('calculating correlation')
                gamma_dx_dz = gamma_coef(all_real_x, all_real_gen)
                
            #     print(gamma_dx_dz)
                print('corr:',gamma_dx_dz)
            # self.corr_scores[self.epoch +1] = float(gamma_dx_dz)

            # prec, recall = get_precision_recall(torch.from_numpy(all_real_x), torch.from_numpy(all_real_gen))
            # print('precision:', prec)
            # print('recall:', recall)
            # self.precision_scores[self.epoch + 1] = prec
            # self.recall_scores[self.epoch + 1] = recall
          
            prec_10, recall_10 = get_precision_recall(torch.from_numpy(all_real_x), torch.from_numpy(all_real_gen), nb_nn=[10])
            prec_20, recall_20 = get_precision_recall(torch.from_numpy(all_real_x), torch.from_numpy(all_real_gen), nb_nn=[20])
            prec_50, recall_50 = get_precision_recall(torch.from_numpy(all_real_x), torch.from_numpy(all_real_gen), nb_nn=[50])
            
            
            print('precision:', prec_10, prec_20, prec_50)
            print('recall:', recall_10, recall_20, recall_50)
            self.precision_scores[self.epoch + 1] = prec_10
            self.recall_scores[self.epoch + 1] = recall_10

            return all_real, all_gen,  all_tissue 

    
    def fit(self, train_data ,test_data, epochs, val=True):

        self.build_CVAE()
        if self.isTrain:
            self.init_train()

        for epoch in range(epochs):

    
            train_loss = 0.0
            print('Epoch: ', epoch)
            self.epoch = epoch
            print('----------Training----------')
            for i, data in enumerate(train_data):
                
                x_GE = data[0].to(self.device)
                x_cat = data[1].to(self.device)
                x_cat_2 = data[2].to(self.device)
                #print(x_GE.shape)
                #print(x_num.shape)
                self.train(x_GE, x_cat,x_cat_2)

                if i==0:
                    batch_losses  =  self.batch_losses
                    
                else:
                    batch_losses = batch_losses + (self.batch_losses)
                
                if (i+1) % self.freq_print == 0:
                    
                    print('[Epoch %d/%d] [Batch %d/%d] [loss : %f] [kl loss : %f]' '[rec loss : %f]' 
                            %(epoch+1, epochs,       # [Epoch -]
                            i+1,len(train_data),   # [Batch -]
                            self.loss_elbo.item(),       # [D loss -]
                            - self.CVAE_model.kl.item(),   
                            self.CVAE_model.reconstruction_loss.item(),
                            #loss_GAN.item(),     # [adv -]
                            #loss_cycle.item(),   # [cycle -]
                            ))
                    
            batch_losses = batch_losses/len(train_data)
            self.loss_dict['kl loss'].append(batch_losses[1])
            self.loss_dict['rec real loss'].append(batch_losses[2])
            self.loss_dict['tot loss'].append(batch_losses[0])

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
                        
                    #metrics = compute_evaluation_metrics(data_real, data_gen, all_real, all_gen)
                        
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

            
                            
                            results_dire_run = os.path.join(self.results_dire, f"test_{run}_epoch_{epoch+1}")
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
    parser = argparse.ArgumentParser(description='CVAE')
    parser.add_argument('--dataset_path', type=str, default='/CompanyDatasets/carlos00/all_tcga_100mb', help='path to dataset')
    parser.add_argument('--output_path', type=str, default='/Experiments/carlos00/iciap/vae/conditional_vae', help='path to save the model')
    parser.add_argument('--batch_size', type=int, default=8, help='input batch size for training (default: 8)')
    parser.add_argument('--epochs', type=int, default=500, help='number of epochs to train (default: 500)')
    parser.add_argument('--freq_compute_test', type=int, default=100, help='number of computing test (default: 100)')
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
    model = CVAE(input_dims= n_genes, 
                        encoder_dims = [256, 256],
                        latent_dims= args.latent_dim,
                        vocab_sizes=[num_disease_types, num_primary_sites],
                        beta = 1.0,
                        #generator_dims = [256, 256, 18665],
                        #discriminator_dims = [256, 256, 1],
                        negative_slope = 0.0, is_bn= False,
                        lr = 5e-5, train=True,freq_compute_test=args.freq_compute_test, freq_visualize_test=100, results_dire= args.output_path)
            
        

            
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