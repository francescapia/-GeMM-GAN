import os
import sys
from tqdm import tqdm
from pathlib import Path
import argparse
from datetime import datetime
import numpy as np
from sklearn.decomposition import PCA 
import matplotlib.pyplot as plt

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn import MultiheadAttention
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

# get file path
FILE_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(FILE_PATH, '..'))

from generative_model_utils import *
from unsupervised_metrics import *
from corr_score import * 
from visualization import *
from data_loader import *
from utility_evaluation import UtilityEvaluator
from utility_primary_s_evaluation import UtilityEvaluatorPrimary
from privacy_evaluator import dcr, nndr, retrieval_accuracy
from glob import glob
from vae import VAE_model

import warnings
warnings.filterwarnings("ignore")


class VAE():

    def __init__(
        self, 
        input_dims, 
        latent_dims=64, 
        encoder_dims=[256, 256],
        beta=1.0,
        negative_slope=0.0, 
        is_bn=False,
        lr = 5e-4, 
        optimizer='rms_prop',
        train=True,
        freq_compute_test = 50,
        gradient_clipping=10,
        results_dire = ''
    ):

        self.input_dims = input_dims
        self.latent_dims = latent_dims
        self.encoder_dims = encoder_dims
        self.decoder_dims = encoder_dims[::-1]
        self.beta = beta
        self.gradient_clipping = gradient_clipping
        
        self.negative_slope = negative_slope
        self.is_bn = is_bn
        self.isTrain  = train
        self.n_genes = input_dims
        self.freq_compute_test = freq_compute_test
        self.result_dire = results_dire
        os.makedirs(self.result_dire, exist_ok=True)
        self.results_dire_fig = os.path.join(self.result_dire, 'figures')
        os.makedirs(self.results_dire_fig, exist_ok=True)
        self.lr = lr
        self.optimizer = optimizer
        # Enabling GPU
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print('device:', self.device)
        
        self.loss_dict = {'reconstruction loss': [],
                          'kl loss': []}
        self.corr_scores = {}
        self.corr_dend_scores = {}
        self.precision_scores = {}
        self.recall_scores = {}

    def init_train(self):

        # Optimizers
        if self.optimizer.lower() == 'rms_prop':
            self.optimizer = torch.optim.RMSprop(self.vae.parameters(), lr=self.lr)
        elif self.optimizer.lower() == 'adam':
            self.optimizer = torch.optim.Adam(self.vae.parameters(), lr=self.lr, betas=(.9, .99))
        elif self.optimizer.lower() == 'adamw':
            self.optimizer = torch.optim.AdamW(self.vae.parameters(), lr=self.lr, betas=(.9, .99), weight_decay=0.01)

    def train_one_epoch(self, dataloader: DataLoader, log_every: int = 4):
        self.vae.train()
        loss_to_log = 0
        grad_norm_to_log = 0
        kl_to_log = 0
        reconstruction_to_log = 0
        data_len = len(dataloader)
        for i, rna_seq in enumerate(dataloader):
            rna_seq = rna_seq[0].to(torch.float32).to(self.device)

            self.optimizer.zero_grad()
            _ = self.vae(rna_seq)
            kl_divergence = self.vae.kl
            reconstruction_loss = self.vae.reconstruction_loss
            loss = self.beta * kl_divergence + reconstruction_loss
            loss.backward()
            total_norm = torch.nn.utils.clip_grad_norm_(self.vae.parameters(), self.gradient_clipping)
            self.optimizer.step()
            
            loss_to_log += loss.detach().item()
            kl_to_log += kl_divergence.detach().item()
            reconstruction_to_log += reconstruction_loss.detach().item()
            grad_norm_to_log += total_norm

            if i % log_every == 0 and i > 0:
                loss_to_log /= log_every
                grad_norm_to_log /= log_every
                kl_to_log /= log_every
                reconstruction_to_log /= log_every
                print(f"[Step {i}/{data_len}] Loss: {loss_to_log:.4f}, KL Divergence: {kl_to_log:.4f}, Reconstruciton Loss: {reconstruction_to_log:.4f} Grad Norm: {grad_norm_to_log:.4f}")
                loss_to_log = 0
                grad_norm_to_log = 0
                kl_to_log = 0
                reconstruction_to_log = 0
            
    def evaluate_vae(self, dataloader: DataLoader):
        self.vae.eval()
        total_loss = 0
        total_kl = 0
        total_reconstruction = 0
        with torch.no_grad():
            for rna_seq in dataloader:
                rna_seq = rna_seq[0].to(torch.float32).to(self.device)

                _ = self.vae(rna_seq)
                kl_divergence = self.vae.kl.item()
                reconstruction_loss = self.vae.reconstruction_loss.item()
                
                total_loss += self.beta * kl_divergence + reconstruction_loss
                total_kl += kl_divergence
                total_reconstruction += reconstruction_loss
                
        avg_loss = total_loss / len(dataloader)
        avg_kl = total_kl / len(dataloader)
        avg_reconstruction = total_reconstruction / len(dataloader)
        return avg_loss, avg_kl, avg_reconstruction

    def generate_samples_all(self, data):

        all_real  = []
        all_gen = []

        for i, data in enumerate(data):
                
            x_GE = data[0].to(self.device)
            x_real, x_gen = self.generate_samples(x_GE)
            if i==0:
                all_gen =  x_gen.cpu().detach().numpy()
                all_real = x_real.cpu().detach().numpy()
            else: 
                all_gen = np.append(all_gen,  x_gen.cpu().detach().numpy(), axis=0)
                all_real = np.append(all_real, x_real.cpu().detach().numpy(), axis=0)

        all_real_x = np.vstack(all_real)
        all_real_gen = np.vstack(all_gen)
        
        return all_real_x, all_real_gen

    def generate_samples(self, gene_expression):
        with torch.no_grad():
            self.vae.eval()
            x_real = gene_expression.clone().to(torch.float32)
            z = torch.normal(0, 1, size=(x_real.shape[0], self.latent_dims), device=self.device)
            x_gen = self.vae.decoder(z)

        return x_real, x_gen

    def set_requires_grad(self, nets, requires_grad=False):

        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def make_lr_schedule(self, base_lr, min_lr, warmup_start_lr, warmup_epochs=10, total_epochs=300):
        decay_gamma = (min_lr / base_lr) ** (1 / (total_epochs - warmup_epochs))  # decay factor

        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                return (warmup_start_lr + (base_lr - warmup_start_lr) * (epoch / warmup_epochs)) / base_lr
            else:
                return decay_gamma ** (epoch - warmup_epochs)
        
        return lr_lambda

    def fit(self, train_data, val_data, test_data, epochs, val=True):
        torch.cuda.init()
        self.vae = VAE_model(
            latent_dims=self.latent_dims, 
            input_dims=self.input_dims,
            encoder_dims=self.encoder_dims)
        self.vae.to(self.device)
        total_parameters= sum(p.numel() for p in self.vae.parameters() if p.requires_grad)
        total_non_trainable_parameters= sum(p.numel() for p in self.vae.parameters() if not p.requires_grad)
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
        
        acc_lr_validation= []
        f1_lr_validation = []
        auc_lr_validation= []
        precision_val = []
        recall_val = []

        lr_scheduler_fn = self.make_lr_schedule(
            base_lr=self.lr, 
            min_lr=max(1e-7, self.lr / 100), 
            warmup_start_lr=1e-7, 
            warmup_epochs=10, 
            total_epochs=epochs)
        scheduler = LambdaLR(self.optimizer, lr_lambda=lr_scheduler_fn)
        for epoch in range(epochs):

            self.epoch = epoch
            print('----------Training Epoch %d/%d----------'%(epoch+1, epochs))

            self.train_one_epoch(train_data, log_every=4)

            loss, kl, reconstruction = self.evaluate_vae(val_data)
            print(f"[Validation] Loss: {loss:.4f}, KL Divergence: {kl:.4f}, Reconstruction Loss: {reconstruction:.4f}")

            scheduler.step()

            if val:       
                
                if (epoch+1) % self.freq_compute_test == 0:
                    self.vae.eval()

                    data_real, data_gen = self.generate_samples_all(train_data)
                    all_real, all_gen = self.generate_samples_all(val_data)
                                                 
                    results_detection = detection(data_real, data_gen, all_real, all_gen)
                    
                    for model_name in results_detection:
                        if model_name == 'Logistic Regression':
                            acc_lr_validation.append(results_detection[model_name]['accuracy'][0])
                            f1_lr_validation.append(results_detection[model_name]['f1'][0])
                            auc_lr_validation.append(results_detection[model_name]['auc'][0])
                            
                    metrics = compute_evaluation_metrics(data_real, data_gen, all_real, all_gen)
                    precision_val.append(metrics['precision_test'])
                    recall_val.append(metrics['recall_test'])

                    all_results = {}
                    all_detection = {}
                    
                    if (epoch+1) == epochs:
                        print('plot umap....')
                        #plot_umaps(all_real, all_gen, self.results_dire_fig, epoch+1, all_tissue,  n_neighbors=300)

                        torch.save(self.vae.state_dict(), os.path.join(self.result_dire,'vae.pt'))
                        print("Models saved at last epoch.")

                        print("Plot of the training metrics")

                        epochs = [i * self.freq_compute_test for i in range(1, len(acc_lr_validation)+1)]

                        plt.figure(figsize=(12, 6))

                        plt.plot(epochs, acc_lr_validation, label='Accuracy (LR)', marker='o')
                        plt.plot(epochs, f1_lr_validation, label='F1 Score (LR)', marker='s')
                        plt.plot(epochs, auc_lr_validation, label='AUC (LR)', marker='^')
                        plt.plot(epochs, precision_val, label='Precision', linestyle='--', marker='d')
                        plt.plot(epochs, recall_val, label='Recall', linestyle='--', marker='x')

                        plt.xlabel('Epoch')
                        plt.ylabel('Metric Value')
                        plt.title('Validation Metrics Over Epochs')
                        plt.legend()
                        plt.grid(True)
                        plt.tight_layout()

                        #plt.savefig("validation_metrics.png")
                        plt.savefig(os.path.join(self.result_dire, 'validation_metrics.png'))
                        plt.close()

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
                            data_real, data_gen = self.generate_samples_all(train_data)
                            all_real, all_gen = self.generate_samples_all(test_data)
                        
                            # must save the data for final utility evaluation    
                            results_dire_run = os.path.join(self.result_dire, f"test_{run}_epoch_{epoch+1}")
                            create_folder(results_dire_run)
                            save_numpy(results_dire_run + '/data_real.npy', data_real)
                            save_numpy(results_dire_run + '/data_gen.npy', data_gen)
                            save_numpy(results_dire_run + '/test_real.npy', all_real)
                            save_numpy(results_dire_run + '/test_gen.npy', all_gen)

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
    parser = argparse.ArgumentParser(description="Train a VAEmodel")
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--num_epochs', type=int, default=1000, help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--latent_dim', type=int, default=64, help='Latent dimension for the model')
    parser.add_argument('--hidden_dim', type=int, default=256, help='Hidden dimension for the model')
    parser.add_argument('--dataset_path', type=str, default='/CompanyDatasets/carlos00/all_tcga_100mb', help='Path to the dataset')
    parser.add_argument('--output_path', type=str, default='/Experiments/carlos00/iciap/conditional_gan', help='Path to save the model')
    parser.add_argument('--num_workers', type=int, default=16, help='Number of workers for data loading')
    parser.add_argument('--freq_compute_test', type=int, default=100, help='Frequency of validation performance')
    parser.add_argument('--optimizer', type=str, default='rms_prop', help='Optimizer to use for training')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    
    print(f'Arguments: {args.__dict__}')
    
    train_loader, val_loader, test_loader, n_genes = dataloader_tcga(
        dataset_path=Path(args.dataset_path),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed)

    h = args.hidden_dim
    model = VAE(
        input_dims= n_genes, 
        latent_dims=args.latent_dim,
        encoder_dims=[h, h],
        optimizer=args.optimizer,
        negative_slope=0.0, is_bn=False,
        lr=5e-5, freq_compute_test=args.freq_compute_test, 
        results_dire=args.output_path)                

    d_l = model.fit(train_loader, val_loader, test_loader, epochs=args.num_epochs)

    print()
    print("--------- Privacy Evaluation ----------")

    def load_data(folder):
        return {
            "data_real": np.load(os.path.join(folder, 'data_real.npy')),
            "data_gen": np.load(os.path.join(folder, 'data_gen.npy')),
            "test_real": np.load(os.path.join(folder, 'test_real.npy')),
            "test_gen": np.load(os.path.join(folder, 'test_gen.npy')),
        }

    results_dirs = sorted(glob(os.path.join(args.output_path, 'test_*')))
    dcr_scores = []
    nndr_scores = []
    for results_dir in results_dirs:
        data = load_data(results_dir)
        dcr_score = dcr(data['data_real'], data['data_gen'], data['test_real'])
        nndr_score = nndr(data['data_real'], data['data_gen'], data['test_real'])
        dcr_scores.append(dcr_score)
        nndr_scores.append(nndr_score)
    
    mean_dcr = np.mean(dcr_scores)
    std_dcr = np.std(dcr_scores)
    mean_nndr = np.mean(nndr_scores)
    std_nndr = np.std(nndr_scores)
    print(f"DCR {mean_dcr:.4f}±{std_dcr:.4f}, NNDR {mean_nndr:.4f}±{std_nndr:.4f}")