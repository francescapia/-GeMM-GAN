import torch
from torch import nn


def reconstruction_loss(x, x_pred, logscale, MSE=True):

    if not MSE:
        scale = torch.exp(logscale)
        mean = x_pred
        dist = torch.distributions.Normal(mean, scale)

        loss = dist.log_prob(x).sum()
    else:
        loss = nn.functional.mse_loss(x_pred, x)
    
    return loss

def kl_divergence(mu, std):
    return torch.mean(-0.5 * torch.sum(1 + std - mu ** 2 - std.exp(), dim = 1), dim = 0)
