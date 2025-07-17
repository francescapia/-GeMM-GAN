import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error
import os

class CorrelationEvaluator:
    def __init__(self, real_path: str, generated_path: str, base_path: str = "./"):
        self.real_path = real_path
        self.generated_path = generated_path
        self.output_dir = os.path.join(base_path, "correlation_report")
        os.makedirs(self.output_dir, exist_ok=True)

    def _load_data(self):
        self.real_data = np.load(self.real_path)
        self.generated_data = np.load(self.generated_path)
        assert self.real_data.shape[1] == self.generated_data.shape[1], "Feature dimension mismatch!"

    def _compute_correlation_matrices(self):
        self.real_corr = np.corrcoef(self.real_data, rowvar=False)
        self.generated_corr = np.corrcoef(self.generated_data, rowvar=False)

    def _compute_mse(self):
        self.mse = mean_squared_error(self.real_corr.flatten(), self.generated_corr.flatten())

    def _plot_difference_heatmap(self):
        diff = self.real_corr - self.generated_corr
        plt.figure(figsize=(10, 8))
        sns.heatmap(diff, cmap="coolwarm", center=0, square=True, xticklabels=False, yticklabels=False)
        plt.title("Correlation Matrix Difference (Real - Generated)")
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "correlation_diff_heatmap.png"), dpi=300)
        plt.close()

    def evaluate(self):
        print("[*] Loading data...")
        self._load_data()

        print("[*] Computing correlation matrices...")
        self._compute_correlation_matrices()

        print("[*] Calculating MSE...")
        self._compute_mse()

        print(f"[✓] Correlation MSE: {self.mse:.6f}")

        print("[*] Plotting heatmap...")
        # self._plot_difference_heatmap()

        return {
            "mse": self.mse,
            "real_correlation_matrix": self.real_corr,
            "generated_correlation_matrix": self.generated_corr,
            "output_dir": self.output_dir
        }
