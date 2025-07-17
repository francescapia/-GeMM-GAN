import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
import pickle
import logging
from pathlib import Path
import random


def train_vae(model: torch.nn.Module, dataloader: DataLoader, optimizer: torch.optim.Optimizer, 
              device: str, gradient_clipping: float = 10, log_every: int = 10, beta: float = 1.0):
    model.train()
    loss_to_log = 0
    grad_norm_to_log = 0
    kl_to_log = 0
    reconstruction_to_log = 0
    data_len = len(dataloader)
    for i, rna_seq in enumerate(dataloader):
        rna_seq = rna_seq[0].to(device)

        optimizer.zero_grad()
        _ = model(rna_seq)
        kl_divergence = model.kl
        reconstruction_loss = model.reconstruction_loss
        loss = beta * kl_divergence + reconstruction_loss
        loss.backward()
        total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
        optimizer.step()
        
        loss_to_log += loss.detach().item()
        kl_to_log += kl_divergence.detach().item()
        reconstruction_to_log += reconstruction_loss.detach().item()
        grad_norm_to_log += total_norm

        if i % log_every == 0 and i > 0:
            loss_to_log /= log_every
            grad_norm_to_log /= log_every
            kl_to_log /= log_every
            reconstruction_to_log /= log_every
            logging.info(f"[Step {i}/{data_len}] Loss: {loss_to_log:.4f}, KL Divergence: {kl_to_log:.4f}, Reconstruciton Loss: {reconstruction_to_log:.4f} Grad Norm: {grad_norm_to_log:.4f}")
            loss_to_log = 0
            grad_norm_to_log = 0
            kl_to_log = 0
            reconstruction_to_log = 0
            

def evaluate_vae(model: torch.nn.Module, dataloader: DataLoader, device: str, beta: float = 1.0):
    model.eval()
    total_loss = 0
    total_kl = 0
    total_reconstruction = 0
    with torch.no_grad():
        for rna_seq in dataloader:
            rna_seq = rna_seq[0].to(device)

            _ = model(rna_seq)
            kl_divergence = model.kl.item()
            reconstruction_loss = model.reconstruction_loss.item()
            
            total_loss += beta * kl_divergence + reconstruction_loss
            total_kl += kl_divergence
            total_reconstruction += reconstruction_loss
            
    avg_loss = total_loss / len(dataloader)
    avg_kl = total_kl / len(dataloader)
    avg_reconstruction = total_reconstruction / len(dataloader)
    return avg_loss, avg_kl, avg_reconstruction


def standardize(x, mean=None, std=None):

    if mean is None:
        mean = np.mean(x, axis=0)
    if std is None:
        std = np.std(x, axis=0)
    return (x - mean) / std


def min_max(x, max=None, min=None):

    if max is None:
        max = np.max(x, axis=0)
    if min is None:
        std = np.min(x, axis=0)
    return (x - min) / (max-min)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def split_data_train_test(case_ids, train_rate=0.80, seed=42, shuffle=True):
    """
    Split data into train, validation, and test sets 
    :param sample_names: list of sample names
    :param train_rate: percentage of training samples
    :param validation_rate: percentage of validation samples
    :param seed: random seed
    :return: lists of train, validation, and test sample indices
    """
    # Set random seed for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    idxs = case_ids
    n_samples = len(case_ids)

    if shuffle:
        np.random.shuffle(idxs)

    t_tr = int(train_rate  * n_samples)
    t_t = n_samples
    train_idxs = idxs[:t_tr]
    test_idxs = idxs[t_tr:t_t]

    assert len(train_idxs) +  len(test_idxs) == n_samples
    return train_idxs, test_idxs
            
            
def dataloader_vae(
    dataset_path: Path, 
    normalize: bool = True, 
    percentage_to_remove: float = 90, 
    norm_type: str = 'standardize',
    batch_size: int = 8, 
    seed: int = 42,
    num_workers: int = 4,
):

    g = torch.Generator()
    g.manual_seed(seed)
    torch.manual_seed(seed)
    
    print('Loading data.......')
    df_expr = pd.read_parquet(dataset_path / 'rna_seq.parquet')
    
    with open(dataset_path / 'dataset_info.pkl', 'rb') as f:
        infos = pickle.load(f)['data_list']
        case_ids = [info['case_id'] for info in infos]
    
    try:
        df_expr = df_expr.loc[case_ids]
    except KeyError as e:
        logging.error(f"There are some missing case IDs in the provided files.")
        raise e
    
    print('Splitting data......')

    zero_percent = (df_expr == 0).sum() / len(df_expr) * 100
    df_expr = df_expr.loc[:, zero_percent <= percentage_to_remove] # remove genes with more than 90% zeros
    
    n_samples = df_expr.shape[0]
    n_genes = df_expr.shape[1] 

    train_case_ids, test_case_ids = split_data_train_test(case_ids)

    df_expr_train = df_expr.loc[train_case_ids]
    df_expr_test = df_expr.loc[test_case_ids]

  
    gene_expressions_train = df_expr_train.values
    gene_expressions_test = df_expr_test.values
    
    print('Building data loaders......')
    train_dataset = TensorDataset(torch.tensor(gene_expressions_train, dtype=torch.float32))
    test_dataset = TensorDataset(torch.tensor(gene_expressions_test, dtype=torch.float32))
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                              worker_init_fn=seed_worker, generator=g, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             worker_init_fn=seed_worker, generator=g, num_workers=num_workers)
    
    return train_loader, test_loader, n_genes