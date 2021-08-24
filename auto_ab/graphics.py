import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from typing import Union


class Graphics:
    def __init__(self):
        pass

    def plot_distributions(self, save_path: str) -> None:
        """Generate distributions and save plot on given path."""
        bins = np.linspace(-10, 10, 100)
        plt.hist(self.datasets['A']['data'], bins, alpha=0.5, label='control')
        plt.hist(self.datasets['B']['data'], bins, alpha=0.5, label='treatment')
        plt.legend(loc='upper right')
        plt.savefig(save_path)


    def plot_distribution(self, X: Union[np.array], ci: np.array, save_path: str) -> None:
        """Generate distributions and save plot on given path."""
        bins = np.linspace(-10, 10, 100)
        plt.hist(X, bins, alpha=0.9, label='Custom metric distribution')
        plt.vlines(ci, ymin=0, ymax=20, linestyle='--')
        plt.legend(loc='upper right')
        plt.savefig(save_path)
        plt.close()

