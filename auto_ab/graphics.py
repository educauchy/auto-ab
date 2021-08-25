import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Union, Dict


class Graphics:
    def __init__(self) -> None:
        pass

    @staticmethod
    def plot_simulation_log(log_path: str):
        df = pd.read_csv(log_path)
        df_pivot = df.pivot(index='split_rate', columns='increment', values='pval_sign_share')
        plt.figure(figsize=(15, 8))
        plt.title('Simulation log')
        sns.heatmap(df_pivot, cmap='Greens', annot=True)
        plt.show()
        plt.close()

    @staticmethod
    def plot_distributions(X: pd.DataFrame, target: str, groups: str, bins: int = 30) -> None:
        """Plot distributions and save plot on given path."""
        plt.hist(X.loc[X[groups] == 'A', target], bins, alpha=0.5, label='control')
        plt.hist(X.loc[X[groups] == 'B', target], bins, alpha=0.5, label='treatment')
        plt.legend(loc='upper right')
        plt.show()
        plt.close()
    #
    def plot_distribution(self, X: Union[np.array], ci: np.array = None, bins: int = 30) -> None:
        """Generate distributions and save plot on given path."""
        plt.hist(X, bins, alpha=0.9, label='Custom metric distribution')
        if ci is not None:
            plt.vlines(ci, ymin=0, ymax=20, linestyle='--')
        plt.legend(loc='upper right')
        plt.show()
        plt.close()
