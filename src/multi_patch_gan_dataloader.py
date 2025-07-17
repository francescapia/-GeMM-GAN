import numpy as np
import pandas as pd
import random
import torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, TensorDataset
import pickle

class MultiPatchGANDataset(Dataset):
    def __init__(self, case_ids, text_embeddings, patches_path, gene_expressions, disease_types, primary_site, num_patches=256):
        self.patches_path = patches_path
        self.case_ids = case_ids
        self.text_embeddings = text_embeddings
        self.gene_expressions = gene_expressions
        self.disease_types = disease_types
        self.primary_site = primary_site
        self.num_patches = num_patches
  

    def __len__(self):
        return self.text_embeddings.shape[0]

    def __getitem__(self, idx):

        case_id = self.case_ids[idx]
        text_embedding = self.text_embeddings[idx]
        gene_expression = self.gene_expressions[idx]
        disease_type = self.disease_types[idx]
        primary_site = self.primary_site[idx]
        
        patches = np.load(self.patches_path / f'{case_id}.npy')
        if patches.shape[0] > self.num_patches:
            patch_indices = np.random.choice(patches.shape[0], self.num_patches, replace=False)
            patches = patches[patch_indices]
            padding_mask = np.array([False] * self.num_patches, dtype=bool)
        else:
            padding = np.zeros((self.num_patches - patches.shape[0], patches.shape[1]))
            patches = np.concatenate((patches, padding), axis=0)
            padding_mask = np.array([False] * patches.shape[0] + [True] * (self.num_patches - patches.shape[0]), dtype=bool)
            
        text_embedding = torch.tensor(text_embedding, dtype=torch.float32)
        gene_expression = torch.tensor(gene_expression, dtype=torch.float32)
        patches = torch.tensor(patches, dtype=torch.float32)
        padding_mask = torch.tensor(padding_mask, dtype=torch.bool)
        disease_type = torch.tensor(disease_type, dtype=torch.long)
        primary_site = torch.tensor(primary_site, dtype=torch.long)

        return text_embedding, gene_expression, patches, padding_mask, disease_type, primary_site


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


def dataloader_multi_patch_conditional_gan(
    dataset_path: Path, 
    normalize: bool = True, 
    percentage_to_remove: float = 90, 
    norm_type: str = 'standardize',
    num_patches: int = 256,
    batch_size: int = 8, 
    seed: int = 42,
    num_workers: int = 4,
    embedding_dim: int = 256,
    text_embedding_file: str = None,
    patch_embeddings_folder: str = None,
):
    
    if text_embedding_file is None:
        text_embedding_file = f'text_embeddings_contrastive_{embedding_dim}.parquet'
    if patch_embeddings_folder is None:
        patch_embeddings_folder = f'patch_embeddings_contrastive_{embedding_dim}'

    g = torch.Generator()
    g.manual_seed(seed)
    torch.manual_seed(seed)
      
    print('Loading data.......')
    df_expr = pd.read_parquet(dataset_path / 'rna_seq.parquet')
    
    with open(dataset_path / 'case_ids.txt', 'r') as f:
        case_ids = f.read().splitlines()
    case_ids = [case_id.strip() for case_id in case_ids]
    case_ids = set(case_ids)
    
    text_embeddings_df = pd.read_parquet(dataset_path / text_embedding_file)
    img_case_ids = (dataset_path / patch_embeddings_folder).glob('*.npy')
    img_case_ids = [img_case_id.stem for img_case_id in img_case_ids]
    text_case_ids = text_embeddings_df.index.tolist()
    rna_seq_case_ids = df_expr.index.tolist()
    common_case_ids = case_ids.intersection(set(img_case_ids)).intersection(set(text_case_ids)).intersection(set(rna_seq_case_ids))
    case_ids = sorted(list(common_case_ids))
    
    print('Splitting data......')

    zero_percent = (df_expr == 0).sum() / len(df_expr) * 100
    df_expr = df_expr.loc[:, zero_percent <= percentage_to_remove] # remove genes with more than 90% zeros
    
    n_samples = len(case_ids)
    n_genes = df_expr.shape[1] 

    #train, test = split_data_train_test(n_samples)
    train, validation, test = split_data(n_samples)
    train_case_ids = [case_ids[i] for i in train]
    validation_case_ids =[case_ids[i] for i in validation]
    test_case_ids = [case_ids[i] for i in test]

    df_expr_train = df_expr.loc[train_case_ids]
    df_expr_validation = df_expr.loc[validation_case_ids]
    df_expr_test = df_expr.loc[test_case_ids]

    print("saving genes")
    np.save('gene_names.npy', df_expr_train.columns)
    print("saving completed")
    # standardize expression data 
    if normalize:
        print('Normalizing data......')
        if norm_type =='standardize':
            x_expr_mean = np.mean(df_expr_train, axis=0)
            x_expr_std = np.std(df_expr_train, axis=0)
            df_expr_train = standardize(df_expr_train, mean=x_expr_mean, std=x_expr_std).fillna(0)
            df_expr_validation = standardize(df_expr_validation, mean=x_expr_mean, std=x_expr_std).fillna(0)
            df_expr_test = standardize(df_expr_test, mean=x_expr_mean, std=x_expr_std).fillna(0)
    
        if norm_type =='min-max':
            x_expr_max = np.max(df_expr_train, axis=0)
            x_expr_min = np.min(df_expr_train, axis=0)
            df_expr_train = min_max(df_expr_train, max=x_expr_max, min=x_expr_min).fillna(0)
            df_expr_validation =  min_max(df_expr_validation, max=x_expr_max, min=x_expr_min).fillna(0)
            df_expr_test = min_max(df_expr_test, max=x_expr_max, min=x_expr_min).fillna(0)
            
    text_embeddings_train = text_embeddings_df.loc[train_case_ids].values
    text_embeddings_validation = text_embeddings_df.loc[validation_case_ids].values
    text_embeddings_test = text_embeddings_df.loc[test_case_ids].values
    
    gene_expressions_train = df_expr_train.values
    gene_expressions_validation = df_expr_validation.values
    gene_expressions_test = df_expr_test.values


    
    with open(dataset_path / 'metainfos.pkl', 'rb') as f:
        metainfos = pickle.load(f)

    disease_types_train = [metainfos[cid]['disease_type'] for cid in train_case_ids]
    disease_types_validation = [metainfos[cid]['disease_type'] for cid in validation_case_ids]
    disease_types_test = [metainfos[cid]['disease_type'] for cid in test_case_ids]

    unique_disease_types = sorted(set(disease_types_train +  disease_types_validation + disease_types_test))
    disease_type_to_idx = {disease: idx for idx, disease in enumerate(unique_disease_types)}

    encoded_disease_types_train = [disease_type_to_idx[d] for d in disease_types_train]
    encoded_disease_types_validation = [disease_type_to_idx[d] for d in disease_types_validation]
    encoded_disease_types_test = [disease_type_to_idx[d] for d in disease_types_test]
    
    
    primary_sites_train = [metainfos[cid]['primary_site'] for cid in train_case_ids]
    primary_sites_validation = [metainfos[cid]['primary_site'] for cid in validation_case_ids]
    primary_sites_test = [metainfos[cid]['primary_site'] for cid in test_case_ids]

    unique_primary_sites = sorted(set(primary_sites_train +  primary_sites_validation + primary_sites_test))
    primary_site_to_idx = {primary_site: idx for idx, primary_site in enumerate(unique_primary_sites)}

    encoded_primary_sites_train = [primary_site_to_idx[d] for d in primary_sites_train]
    encoded_primary_sites_validation = [primary_site_to_idx[d] for d in primary_sites_validation]
    encoded_primary_sites_test = [primary_site_to_idx[d] for d in primary_sites_test]
    
    print('Building data loaders......')
    train_dataset = MultiPatchGANDataset(train_case_ids, text_embeddings_train, dataset_path / patch_embeddings_folder, 
                                         gene_expressions_train, encoded_disease_types_train, encoded_primary_sites_train, num_patches=num_patches)

    validation_dataset = MultiPatchGANDataset(validation_case_ids, text_embeddings_validation, dataset_path / patch_embeddings_folder, 
                                         gene_expressions_validation, encoded_disease_types_validation, encoded_primary_sites_validation, num_patches=num_patches)
    
    test_dataset = MultiPatchGANDataset(test_case_ids, text_embeddings_test, dataset_path / patch_embeddings_folder,
                                        gene_expressions_test, encoded_disease_types_test, encoded_primary_sites_test, num_patches=num_patches)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                              worker_init_fn=seed_worker, generator=g, num_workers=num_workers)
    
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=True, 
                              worker_init_fn=seed_worker, generator=g, num_workers=num_workers)
    
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             worker_init_fn=seed_worker, generator=g, num_workers=num_workers)
    
    
    return train_loader, validation_loader, test_loader, n_genes