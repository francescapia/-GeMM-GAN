import os
import numpy as np
from glob import glob
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from catboost import CatBoostClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.pyplot as plt
from classifiers.mlp import TorchMLPClassifier
from lightgbm import LGBMClassifier
import warnings
warnings.filterwarnings("ignore")

def plot_embedding_by_tissue(folder, method='umap', save_path=None):
    data = load_data(folder)
    
    # Dati e label
    data_real = data['test_real']
    data_gen = data['test_gen']
    labels = data['test_labels']
    
    # Concatenazione e flag per reale/fake
    combined_data = np.vstack([data_real, data_gen])
    is_generated = np.array([0] * len(data_real) + [1] * len(data_gen))
    combined_labels = np.concatenate([labels, labels])  
    if method == 'umap':
        #reducer = umap.UMAP(random_state=42)
        pass
    elif method == 'tsne':
        reducer = TSNE(n_iter=1000, perplexity=30, random_state=42)
    else:
        raise ValueError("Method must be 'umap' or 'tsne'")
    
    embedding = reducer.fit_transform(combined_data)

    emb_real = embedding[:len(data_real)]
    emb_gen = embedding[len(data_real):]

    # --- Plot 1: Colorato per tessuto ---
    plt.figure(figsize=(10, 8))
    palette = sns.color_palette("hls", len(np.unique(labels)))
    sns.scatterplot(x=emb_real[:, 0], y=emb_real[:, 1], hue=labels,
                    palette=palette, alpha=0.6, s=40, label='Real')
    sns.scatterplot(x=emb_gen[:, 0], y=emb_gen[:, 1], hue=labels,
                    palette=palette, alpha=0.6, s=40, marker='X', label='Generated', legend=False)
    plt.title(f"{method.upper()} - Real vs Generated (Colored by Tissue)")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.legend(title="Tissue", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    if save_path:
        plt.savefig(os.path.join(save_path, f"{method}_by_tissue.png"), dpi=300)
    plt.show()

    # --- Plot 2: Colorato per tipo (reale o generato) ---
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=embedding[:, 0], y=embedding[:, 1],
                    hue=is_generated,
                    palette={0: 'steelblue', 1: 'darkorange'},
                    alpha=0.6, s=40)
    plt.title(f"{method.upper()} - Real vs Generated (Colored by Type)")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.legend(title='Type', labels=['Real', 'Generated'])
    plt.tight_layout()
    if save_path:
        plt.savefig(os.path.join(save_path, f"{method}_by_type.png"), dpi=300)
    plt.show()






def load_data(folder):
    return {
        "data_real": np.load(os.path.join(folder, 'data_real.npy')),
        "data_gen": np.load(os.path.join(folder, 'data_gen.npy')),
        "test_real": np.load(os.path.join(folder, 'test_real.npy')),
        "test_gen": np.load(os.path.join(folder, 'test_gen.npy')),
        "train_labels_real": np.load(os.path.join(folder, 'train_primary_site_real.npy')),
        "train_labels_gen": np.load(os.path.join(folder, 'train_primary_site_gen.npy')),
        "test_labels_real": np.load(os.path.join(folder, 'test_primary_site_real.npy')),
        "test_labels_gen": np.load(os.path.join(folder, 'test_primary_site_gen.npy')),
    }

def compute_metrics(y_true, y_pred, metric_funcs):
    results = {}
    for name, func in metric_funcs.items():
        if name in ['Precision', 'Recall','F1']:  # Queste metriche supportano 'average'
            results[name] = func(y_true, y_pred, average='weighted')
        else:
            results[name] = func(y_true, y_pred)  # Altre metriche che non supportano 'average'
    return results

# Evaluation Class
class UtilityEvaluatorPrimary:
    def __init__(self, results_path):
        self.results_dirs = sorted(glob(os.path.join(results_path, 'test_*')))
        print(f"Found {len(self.results_dirs)} result folders.")

        # self.classifiers = {
        #     'RandomForest': RandomForestClassifier(),
        #  #   'LogisticRegression': LogisticRegression(max_iter=300, solver='saga', penalty='l2'),
        #     #'CatBoost': CatBoostClassifier(verbose=0)
        #     'MLP': MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, random_state=42)
        # }
        self.classifiers = {
            'MLP': TorchMLPClassifier(hidden_dims=[100,], num_epochs=50, random_state=42, verbose=True),
            # 'LogisticRegression': TorchMLPClassifier(hidden_dims=[], num_epochs=100, random_state=42),
            'RandomForest': LGBMClassifier(boosting_type='rf', n_estimators=100, min_child_samples=2, colsample_bytree=0.01, random_state=42, verbose=-1)
        }

        self.metrics = {
            'Accuracy': accuracy_score,
            'F1': f1_score,
            'Precision': precision_score,
            'Recall': recall_score
        }

        self.scores = {
            setting: {clf: {m: [] for m in self.metrics} for clf in self.classifiers}
            for setting in ['TRTR', 'TSTR', 'TR+TSR']
        }

    def evaluate(self):
        for folder in self.results_dirs:
            print(f"\nEvaluating {folder}")
           
            data = load_data(folder)

            for clf_name, clf in self.classifiers.items():
                print(clf_name)
                # TRTR: Train on real, test on synthetic
                print('trtr')
                clf.fit(data['test_real'], data['test_labels_real'])
                preds = clf.predict(data['data_real'])
                for m, val in compute_metrics(data['train_labels_real'], preds, self.metrics).items():
                    self.scores['TRTR'][clf_name][m].append(val)

                # TSTR: Train on synthetic, test on real
                print('tstr')
                clf.fit(data['test_gen'], data['test_labels_gen'])
                preds = clf.predict(data['data_real'])
                for m, val in compute_metrics(data['train_labels_real'], preds, self.metrics).items():
                    self.scores['TSTR'][clf_name][m].append(val)

                # TR+TSR: Train on real + synthetic, test on real
                print('TR+TSR')
                X_train = np.concatenate([data['test_real'], data['test_gen']], axis=0)
                y_train = np.concatenate([data['test_labels_real'], data['test_labels_gen']], axis=0)
                clf.fit(X_train, y_train)
                preds = clf.predict(data['data_real'])
                for m, val in compute_metrics(data['train_labels_real'], preds, self.metrics).items():
                    self.scores['TR+TSR'][clf_name][m].append(val)
                    
                


           
    def report(self):
        for setting in self.scores:
            print(f'\n--- {setting} ---')
            for clf_name in self.scores[setting]:
                print(f'\nClassifier: {clf_name}')
                for m in self.metrics:
                    values = self.scores[setting][clf_name][m]
                    mean = np.mean(values)
                    std = np.std(values)
                    print(f'{m}: {mean:.4f} ± {std:.4f}')
                    
                    
