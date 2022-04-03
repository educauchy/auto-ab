import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Union, Dict, List
from scipy.stats import norm, mode


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
        a = X.loc[X[groups] == 'A', target]
        b = X.loc[X[groups] == 'B', target]
        a_mean = a.mean()
        b_mean = b.mean()
        plt.hist(a, bins, alpha=0.5, label='control')
        plt.hist(b, bins, alpha=0.5, label='treatment')
        plt.vlines([a_mean, b_mean], ymin=0, ymax=1000, linestyle='--')
        plt.legend(loc='upper right')
        plt.show()
        plt.close()

    def plot_distribution(self, X: Union[np.array], ci: np.array = None, bins: int = 30) -> None:
        """Generate distributions and save plot on given path."""
        plt.hist(X, bins, alpha=0.9, label='Custom metric distribution')
        if ci is not None:
            plt.vlines(ci, ymin=0, ymax=20, linestyle='--')
        plt.legend(loc='upper right')
        plt.show()
        plt.close()

    @staticmethod
    def plot_mean_experiment(a: Union[np.array, List[Union[int, float]]] = None,
                            b: Union[np.array, List[Union[int, float]]] = None,
                            alternative: str = 'two-sided',
                            metric: str = 'mean', alpha: float = 0.05, beta: float = 0.2) -> None:
        bins = 50
        a_mean = np.mean(a)
        b_mean = np.mean(b)
        threshold = np.quantile(a, 0.975)
        fig, ax = plt.subplots(figsize=(20, 12))
        ax.text(a_mean, 100, 'H0', fontsize='xx-large')
        ax.text(b_mean, 100, 'H1', fontsize='xx-large')
        ax.hist(a, bins, alpha=0.5, label='control', color='Red')
        ax.hist(b, bins, alpha=0.5, label='treatment', color='Green')
        ax.axvline(x=a_mean, color='Red')
        ax.axvline(x=b_mean, color='Green')
        ax.axvline(x=threshold, color='Blue', label='critical value')
        ax.legend()
        plt.show()

    @staticmethod
    def plot_bootstrap_confint(X: Union[np.array, List[Union[int, float]]] = None,
                               alternative: str = 'two-sided',
                               alpha: float = 0.05, beta: float = 0.2
                               ) -> None:
        bins = 50
        threshold = np.quantile(X, 0.025)
        fig, ax = plt.subplots(figsize=(20, 12))
        ax.hist(X, bins, alpha=0.5, label='Differences in metric', color='Red')
        ax.axvline(x=0, color='Red', label='No difference')
        ax.axvline(x=threshold, color='Blue', label='critical value')
        ax.legend()
        plt.show()
        pass

if __name__ == '__main__':
    a = np.random.normal(0, 4, 5_000)
    b = np.random.normal(0, 6, 5_000)
    gr = Graphics()
    # gr.plot_mean_experiment(a, b)

    metric_diffs: List[float] = []
    for _ in range(5000):
        x_boot = np.random.choice(a, size=a.shape[0], replace=True)
        y_boot = np.random.choice(b, size=b.shape[0], replace=True)
        metric_diffs.append(np.mean(y_boot) - np.mean(x_boot))
    gr.plot_bootstrap_confint(metric_diffs)
