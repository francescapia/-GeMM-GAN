import torch
import numpy as np
from rnaseq_contrastive_model import retrieve_cross_modal
from tqdm import tqdm
from glob import glob
import os


def dcr(real_data, gen_data, test_data, batch_size=128):
    real_data_th = torch.tensor(real_data).cuda()
    syn_data_th = torch.tensor(gen_data).cuda()
    test_data_th = torch.tensor(test_data).cuda()

    dcrs_real = []
    dcrs_test = []

    for i in range((gen_data.shape[0] // batch_size) + 1):
        if i != (gen_data.shape[0] // batch_size):
            batch_syn_data_th = syn_data_th[i*batch_size: (i+1) * batch_size]
        else:
            batch_syn_data_th = syn_data_th[i*batch_size:]
            
        dcr_real = (batch_syn_data_th[:, None] - real_data_th).pow(2).sum(dim=2).sqrt().min(dim=1).values
        dcr_test = (batch_syn_data_th[:, None] - test_data_th).pow(2).sum(dim=2).sqrt().min(dim=1).values
        dcrs_real.append(dcr_real)
        dcrs_test.append(dcr_test)
        
    dcrs_real = torch.cat(dcrs_real)
    dcrs_test = torch.cat(dcrs_test)

    score = (dcrs_real < dcrs_test).nonzero().shape[0] / dcrs_real.shape[0]
    return score

def nndr(real_data, gen_data, test_data, batch_size=128):
    real_data_th = torch.tensor(real_data).cuda()
    syn_data_th = torch.tensor(gen_data).cuda()
    test_data_th = torch.tensor(test_data).cuda()

    nndrs_real = []
    nndrs_test = []

    for i in range((gen_data.shape[0] // batch_size) + 1):
        if i != (gen_data.shape[0] // batch_size):
            batch_syn_data_th = syn_data_th[i*batch_size: (i+1) * batch_size]
        else:
            batch_syn_data_th = syn_data_th[i*batch_size:]

        distances_real = (batch_syn_data_th[:, None] - real_data_th).pow(2).sum(dim=2).sqrt()
        distances_real_sorted, _ = torch.sort(distances_real, dim=1)
        first_neighbors_real = distances_real_sorted[:, 0]
        second_neighbors_real = distances_real_sorted[:, 1]
        nndr_real = first_neighbors_real / second_neighbors_real
        nndrs_real.append(nndr_real)

        distances_test = (batch_syn_data_th[:, None] - test_data_th).pow(2).sum(dim=2).sqrt()
        distances_test_sorted, _ = torch.sort(distances_test, dim=1)
        first_neighbors_test = distances_test_sorted[:, 0]
        second_neighbors_test = distances_test_sorted[:, 1]
        nndr_test = first_neighbors_test / second_neighbors_test
        nndrs_test.append(nndr_test)
        
    nndrs_real = torch.cat(nndrs_real)
    nndrs_test = torch.cat(nndrs_test)

    score = (nndrs_real < nndrs_test).nonzero().shape[0] / nndrs_real.shape[0]
    return score

def retrieval_accuracy(real_embeddings, gen_embeddings, real_labels, gen_labels, real_patch_embeddings, real_text_embeddings, batch_size=128):
    '''
    Compute the retrieval accuracy of the generated data.
    Args:
        real_embeddings (Tensor): Real data. Shape: (N, D)
        gen_embeddings (Tensor: Generated data. Shape: (M, D)
        real_labels (Tensor: Labels of the real data. Shape: (N, )
        gen_labels (Tensor): Labels of the generated data. Shape: (M, )
        real_patches (list[Tensor]): Patches of the real data. List of N tensors, each of shape (P_i, D)
        real_text (Tensor): Text of the real data. Shape: (N, D)
    Returns:
        float: Retrieval accuracy.
    '''

    patches_labels = [[real_labels[i].item()] * patches.shape[0] for i, patches in enumerate(real_patch_embeddings)]
    patches_labels = torch.tensor(sum(patches_labels, []))
    patches_embeddings = torch.cat(real_patch_embeddings)

    indices = []
    for i in range((gen_embeddings.shape[0] // batch_size) + 1):
        if i != (gen_embeddings.shape[0] // batch_size):
            batch_syn_data_th = gen_embeddings[i*batch_size: (i+1) * batch_size]
        else:
            batch_syn_data_th = gen_embeddings[i*batch_size:]
        batch_indices, batch_scores = retrieve_cross_modal(batch_syn_data_th, patches_embeddings, top_k=1)
        batch_indices = batch_indices.squeeze(1)
        indices.append(batch_indices)

    indices = torch.cat(indices, dim=0).cpu()
    labels_retrieved = patches_labels[indices]
    accuracy_image = (labels_retrieved == gen_labels.cpu()).sum().item() / len(gen_labels)

    indices = []
    for i in range((gen_embeddings.shape[0] // batch_size) + 1):
        if i != (gen_embeddings.shape[0] // batch_size):
            batch_syn_data_th = gen_embeddings[i*batch_size: (i+1) * batch_size]
        else:
            batch_syn_data_th = gen_embeddings[i*batch_size:]
        batch_indices, batch_scores = retrieve_cross_modal(batch_syn_data_th, real_text_embeddings, top_k=1)
        batch_indices = batch_indices.squeeze(1)
        indices.append(batch_indices)

    indices = torch.cat(indices, dim=0).cpu()
    labels_retrieved = real_labels[indices].cpu()
    accuracy_text = (labels_retrieved == gen_labels.cpu()).sum().item() / len(gen_labels)

    return accuracy_image, accuracy_text


def load_data(folder):
    return {
        "data_real": np.load(os.path.join(folder, 'data_real.npy')),
        "data_gen": np.load(os.path.join(folder, 'data_gen.npy')),
        "test_real": np.load(os.path.join(folder, 'test_real.npy')),
        "test_gen": np.load(os.path.join(folder, 'test_gen.npy')),
    }


class PrivacyEvaluator:

    def __init__(self, results_path):
        self.results_dirs = sorted(glob(os.path.join(results_path, 'test_*')))
        print(f"Found {len(self.results_dirs)} result folders.")

        self.scores = {
            'dcr': [],
            'nndr': [],
        }

    def _dcr(self, real_data, gen_data, batch_size=128):
        real_data_th = torch.tensor(real_data).cuda()
        syn_data_th = torch.tensor(gen_data).cuda()

        dcrs_real = []

        for i in range((gen_data.shape[0] // batch_size) + 1):
            if i != (gen_data.shape[0] // batch_size):
                batch_syn_data_th = syn_data_th[i*batch_size: (i+1) * batch_size]
            else:
                batch_syn_data_th = syn_data_th[i*batch_size:]
                
            dcr_real = (batch_syn_data_th[:, None] - real_data_th).pow(2).sum(dim=2).sqrt().min(dim=1).values
            dcrs_real.append(dcr_real)
            
        return torch.cat(dcrs_real).mean()

    def _nndr(self, real_data, gen_data, batch_size=128):
        real_data_th = torch.tensor(real_data).cuda()
        syn_data_th = torch.tensor(gen_data).cuda()

        nndrs_real = []

        for i in range((gen_data.shape[0] // batch_size) + 1):
            if i != (gen_data.shape[0] // batch_size):
                batch_syn_data_th = syn_data_th[i*batch_size: (i+1) * batch_size]
            else:
                batch_syn_data_th = syn_data_th[i*batch_size:]

            distances_real = (batch_syn_data_th[:, None] - real_data_th).pow(2).sum(dim=2).sqrt()
            distances_real_sorted, _ = torch.sort(distances_real, dim=1)
            first_neighbors_real = distances_real_sorted[:, 0]
            second_neighbors_real = distances_real_sorted[:, 1]
            nndr_real = first_neighbors_real / second_neighbors_real
            nndrs_real.append(nndr_real)
            
        return torch.cat(nndrs_real).mean()

    def evaluate(self):
        for folder in self.results_dirs:
            print(f"\nEvaluating {folder}")
           
            data = load_data(folder)
            nndr_score = self._nndr(data['data_real'], data['data_gen'])
            dcr_score = self._dcr(data['data_real'], data['data_gen'])
            self.scores['nndr'].append(nndr_score.item())
            self.scores['dcr'].append(dcr_score.item())

    def report(self):
        print("\nPrivacy Evaluation Results:")
        print(f"DCR: {np.mean(self.scores['dcr']):.4f} ± {np.std(self.scores['dcr']):.4f}")
        print(f"NNDR: {np.mean(self.scores['nndr']):.4f} ± {np.std(self.scores['nndr']):.4f}")