import numpy as np 
import random
import torch
import torch.nn as nn

# Set seed
SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)
random.seed(SEED)


def build_encoder(input_dims, encoder_dims, negative_slope= 0.0, is_bn=False):

    encoder = []
    for i in range(len(encoder_dims)):
        if i == 0:
            encoder.append(build_linear_block(input_dims, encoder_dims[i], negative_slope= negative_slope, is_bn=is_bn))
        else:
            encoder.append(build_linear_block(encoder_dims[i-1], encoder_dims[i], negative_slope= negative_slope, is_bn=is_bn))
    encoder = nn.Sequential(*encoder)
    return encoder



def build_decoder(latent_dims, decoder_dims, input_dims, negative_slope= 0.0, is_bn=False):

    decoder = []
    for i in range(len(decoder_dims)):
        if i == 0:
            decoder.append(build_linear_block(latent_dims, decoder_dims[i], negative_slope= negative_slope, is_bn=is_bn))
        else:
            decoder.append(build_linear_block(decoder_dims[i-1], decoder_dims[i], negative_slope= negative_slope, is_bn=is_bn))
    
    decoder.append(nn.Linear(decoder_dims[-1], input_dims))
    decoder = nn.Sequential(*decoder)


    return decoder


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
            nn.LeakyReLU(negative_slope=negative_slope)
        )

    return net

def categorical_embedding(vocab_sizes):

    #n_cat_vars = len(vocab_sizes)
    embedder = nn.ModuleList()
    for vs in vocab_sizes:
        emdedding_dims = int(vs**0.5) +1 
        embedder.append(nn.Embedding(vs,  emdedding_dims))
    return embedder
