import torch
import torch.nn as nn

import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)

from model_utils_vae import *
from losses import *
# set seed
SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)
random.seed(SEED)


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
        
        x_encoded = self.encoder(x)
        mu, log_var = self.mu(x_encoded), self.log_var(x_encoded)
        #torch.exp(0.5 * logvar)
        q = torch.distributions.Normal(0,1)
        z = mu + torch.exp(log_var*0.5)* q.rsample(mu.shape).to(self.device)
        # q = torch.distributions.Normal(mu, torch.exp(log_var))
        # z = q.rsample(mu.shape)

        self.kl = kl_divergence(mu, torch.exp(log_var))

        x_pred = self.decoder(z)
        
        self.reconstruction_loss = reconstruction_loss(x, x_pred, self.log_scale)
        return x_pred
    
    def to(self, device):
        self.encoder.to(device)
        self.decoder.to(device)
        self.mu.to(device)
        self.log_var.to(device)
        self.log_scale.to(device)
        self.device = device
        return self
    
    
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
        # self.numerical_dims = len(numerical_dims)
        self.negative_slope = negative_slope
        self.vocab_sizes = vocab_sizes
        self.n_cat_vars = len(self.vocab_sizes)
        self.categorical_embedded_dims = sum([int(vs**0.5)+1 for vs in self.vocab_sizes])
        self.input_dec_dims = self.latent_dims +  self.categorical_embedded_dims  #+ self.numerical_dims +  self.categorical_embedded_dims 
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

    def decode(self, z,x_cat): #, x_num:
        
        embedded_cat_vars = []
        #cat_x = self.embedding(categorical_covariates)
        for i, module in enumerate(self.categorical_embedding):
            cat_x = x_cat[:,i]
            embedded_cat_vars.append(module(cat_x))
        if self.n_cat_vars == 1:
            embedded_cat_vars = embedded_cat_vars[0]

        else:
            embedded_cat_vars = torch.cat(embedded_cat_vars,dim=1)

        #z = torch.randn(x_num.shape[0], self.latent_dims)
        
        #x_pred = torch.cat((z, x_num, x_cat), axis=1)
        #x_pred = torch.cat((z, x_num, embedded_cat_vars), axis=1)    
        x_pred = torch.cat((z, embedded_cat_vars), axis=1)
        for module in self.decoder:
            x_pred = module(x_pred)
        return x_pred

    def reparameterize(self, mu, log_var):
    
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, x, x_cat): #, x_num, x_cat):
        
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
            cat_x = x_cat[:,i]
            embedded_cat_vars.append(module(cat_x))
        if self.n_cat_vars == 1:
            embedded_cat_vars = embedded_cat_vars[0]
    
        else:
            embedded_cat_vars = torch.cat(embedded_cat_vars,dim=1)

        x_pred = torch.cat((z, embedded_cat_vars), axis=1)
        
        for module in self.decoder:
            x_pred = module(x_pred)
        
        self.reconstruction_loss = reconstruction_loss(x, x_pred, self.log_scale)
        return x_pred
    
    
