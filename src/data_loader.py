import os
import numpy as np
import pandas as pd
import random
import torch
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path



def split_data(n_samples, train_rate=0.80, validation_rate=0.20, seed=42, shuffle=True):
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
    idxs = np.arange(n_samples)

    if shuffle:
        np.random.shuffle(idxs)
    t_tr = int(train_rate * (1-validation_rate) * n_samples)
    t_val = t_tr  + int(train_rate * validation_rate * n_samples)
    #t_t = t_val + int((1-train_rate)*n_samples)
    t_t = n_samples
    train_idxs = idxs[:t_tr]
    validation_idsx = idxs[t_tr:t_val]
    test_idxs = idxs[t_val:t_t]
    #print(train_idxs.shape[0] + validation_idsx.shape[0]+  test_idxs.shape[0])
    assert train_idxs.shape[0] + validation_idsx.shape[0]+  test_idxs.shape[0] == n_samples
    return train_idxs, validation_idsx, test_idxs

def split_data_train_test(n_samples, train_rate=0.80, seed=42, shuffle=True):
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
    idxs = np.arange(n_samples)

    if shuffle:
        np.random.shuffle(idxs)
    t_tr = int(train_rate  * n_samples)
    t_t = n_samples
    train_idxs = idxs[:t_tr]
    test_idxs = idxs[t_tr:t_t]

    assert train_idxs.shape[0] +  test_idxs.shape[0] == n_samples
    return train_idxs, test_idxs



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

def dataloader_tcga(
    dataset_path: Path,
    normalize: bool = True, 
    percentage_to_remove: float = 90, 
    norm_type: str = 'standardize',
    batch_size: int = 8, 
    seed: int = 42,
    num_workers: int = 4):

    g = torch.Generator()
    g.manual_seed(seed)
    torch.manual_seed(seed)
  

    print('Loading data.......')
    df_expr = pd.read_parquet(dataset_path / 'rna_seq.parquet')   
    
    with open(dataset_path / 'case_ids.txt', 'r') as f:
        case_ids = f.read().splitlines()
    case_ids = [case_id.strip() for case_id in case_ids]
    case_ids = set(case_ids)
    
    # this is just to reproduce the same split as in the contrastive training and conditional gan training
    text_embeddings_df = pd.read_parquet(dataset_path / 'text_embeddings_contrastive_256.parquet')
    img_case_ids = (dataset_path / 'patch_embeddings_contrastive_256').glob('*.npy')
    img_case_ids = [img_case_id.stem for img_case_id in img_case_ids]
    text_case_ids = text_embeddings_df.index.tolist()
    rna_seq_case_ids = df_expr.index.tolist()
    common_case_ids = case_ids.intersection(set(img_case_ids)).intersection(set(text_case_ids)).intersection(set(rna_seq_case_ids))
    case_ids = sorted(list(common_case_ids))
    
    del text_embeddings_df
    
    print('Splitting data......')

    zero_percent = (df_expr == 0).sum() / len(df_expr) * 100
    df_expr = df_expr.loc[:, zero_percent <= percentage_to_remove] # remove genes with more than 90% zeros
    
    n_samples = len(case_ids)
    n_genes = df_expr.shape[1] 

    #train, test = split_data_train_test(n_samples)
    train, validation, test = split_data(n_samples)
    
    train_case_ids = [case_ids[i] for i in train]
    validation_case_ids = [case_ids[i] for i in validation]
    test_case_ids = [case_ids[i] for i in test]

    df_expr_train = df_expr.loc[train_case_ids]
    df_expr_validation = df_expr.loc[validation_case_ids]
    df_expr_test = df_expr.loc[test_case_ids]

    # standardize expression data 
    if normalize:
        print('Normalizing data......')
        if norm_type =='standardize':
            x_expr_mean = np.mean(df_expr_train, axis=0)
            x_expr_std = np.std(df_expr_train, axis=0)
            df_expr_train = standardize(df_expr_train, mean=x_expr_mean, std=x_expr_std).fillna(0)
            df_expr_test = standardize(df_expr_test, mean=x_expr_mean, std=x_expr_std).fillna(0)
            df_expr_validation = standardize(df_expr_validation, mean=x_expr_mean, std=x_expr_std).fillna(0)
    
 
        if norm_type =='min-max':
            x_expr_max = np.max(df_expr_train, axis=0)
            x_expr_min = np.min(df_expr_train, axis=0)
            df_expr_train = min_max(df_expr_train, max=x_expr_max, min=x_expr_min).fillna(0)
            df_expr_test = min_max(df_expr_test, max=x_expr_max, min=x_expr_min).fillna(0)
            df_expr_validation = min_max(df_expr_validation, max=x_expr_max, min=x_expr_min).fillna(0)


    print('Building data loaders......')
    # build loaders
    train_loader = DataLoader(TensorDataset(torch.from_numpy(df_expr_train.to_numpy())), 
                                            batch_size=batch_size, shuffle=True,  worker_init_fn=seed_worker,
                                            generator=g, num_workers=num_workers)

    validation_loader = DataLoader(TensorDataset(torch.from_numpy(df_expr_validation.to_numpy())), 
                                            batch_size=batch_size, shuffle=True,  worker_init_fn=seed_worker,
                                            generator=g, num_workers=num_workers)
      
    test_loader =  DataLoader(TensorDataset(torch.from_numpy(df_expr_test.to_numpy())), 
                                            batch_size=batch_size, shuffle=False, worker_init_fn=seed_worker,
                                            generator=g, num_workers=num_workers)
    
    print('Loading Completed!')
    
    return train_loader, validation_loader, test_loader, n_genes


def dataloader_tcga_cond(df_expr, df_emb,
                         normalize=True, percentage_to_remove=90, norm_type='standardize',
                         batch_size=4, seed=42):
    g = torch.Generator()
    g.manual_seed(seed)
    torch.manual_seed(seed)

    print('Loading data.......')

    # Join based on index (sample ID)
    df_expr = df_expr.copy()
    df_emb = df_emb.copy()

    # Allinea gli indici
    df_expr = df_expr.loc[df_expr.index.intersection(df_emb.index)]
    df_emb = df_emb.loc[df_emb.index.intersection(df_expr.index)]

    print('Splitting data......')

    # Remove genes with more than 'percentage_to_remove' zeros
    zero_percent = (df_expr == 0).sum() / len(df_expr) * 100
    df_expr = df_expr.loc[:, zero_percent <= percentage_to_remove]  # Remove genes with >90% zeros
    
    n_samples = df_expr.shape[0]
    n_genes = df_expr.shape[1]

    # Train/test split
   # train_idx, test_idx = split_data_train_test(n_samples)
    train_idx, validation_idx, test_idx = split_data(n_samples)
    df_expr_train, df_emb_train = df_expr.iloc[train_idx], df_emb.iloc[train_idx]
    df_expr_validation, df_emb_validation = df_expr.iloc[validation_idx], df_emb.iloc[validation_idx]
    df_expr_test, df_emb_test = df_expr.iloc[test_idx], df_emb.iloc[test_idx]

    # Standardize or normalize the expression data
    if normalize:
        print('Normalizing data......')
        if norm_type == 'standardize':
            x_expr_mean = np.mean(df_expr_train, axis=0)
            x_expr_std = np.std(df_expr_train, axis=0)
            df_expr_train = standardize(df_expr_train, mean=x_expr_mean, std=x_expr_std).fillna(0)
            df_expr_validation = standardize(df_expr_validation, mean=x_expr_mean, std=x_expr_std).fillna(0)
            df_expr_test = standardize(df_expr_test, mean=x_expr_mean, std=x_expr_std).fillna(0)

        elif norm_type == 'min-max':
            x_expr_max = np.max(df_expr_train, axis=0)
            x_expr_min = np.min(df_expr_train, axis=0)
            df_expr_train = min_max(df_expr_train, max=x_expr_max, min=x_expr_min).fillna(0)
            df_expr_validation = min_max(df_expr_validation, max=x_expr_max, min=x_expr_min).fillna(0)
            df_expr_test = min_max(df_expr_test, max=x_expr_max, min=x_expr_min).fillna(0)

    print('Building data loaders......')

    # Since df_emb already has float values, just convert it to numpy arrays
    df_emb_train_np = df_emb_train.to_numpy(dtype=np.float32)
    df_emb_validation_np = df_emb_validation.to_numpy(dtype=np.float32)
    df_emb_test_np = df_emb_test.to_numpy(dtype=np.float32)

    print('Embedding shape for train:', df_emb_train_np.shape)
    print('Embedding shape for test:', df_emb_test_np.shape)

    # Build DataLoaders
    train_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(df_expr_train.to_numpy()).float(),
            torch.from_numpy(df_emb_train_np).float()
        ),
        batch_size=batch_size, shuffle=True, worker_init_fn=seed_worker, generator=g
    )

    validation_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(df_expr_validation.to_numpy()).float(),
            torch.from_numpy(df_emb_validation_np).float()
        ),
        batch_size=batch_size, shuffle=True, worker_init_fn=seed_worker, generator=g
    )

    test_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(df_expr_test.to_numpy()).float(),
            torch.from_numpy(df_emb_test_np).float()
        ),
        batch_size=batch_size, shuffle=False, worker_init_fn=seed_worker, generator=g
    )

    print('Loading Completed!')

    return train_loader, validation_loader, test_loader, n_genes