import numpy as np
from random import randint
import pandas as pd
from numpy.random import normal, binomial
import statsmodels.stats.api as sms
import math
from collections import Counter
from scipy.stats import mannwhitneyu
import matplotlib.pyplot as plt
import os


def generate_distribution(dist: str, params: tuple, n_samples: int) -> np.array:
    """Return distribution with given parameters and number of samples."""
    if dist == 'normal':
        return normal(*params, n_samples)
    elif dist == 'binomial':
        return binomial(*params, n_samples)

def read_file(path: str) -> pd.DataFrame:
    """Read file and return pandas dataframe"""
    _, file_ext = os.path.splitext(path)
    if file_ext == '.csv':
        df = pd.read_csv(path, encoding='utf8')
    elif file_ext == '.xls' or file_ext == '.xlsx':
        df = pd.read_excel(path, encoding='utf8')
    return df


class ABtest:
    """Perform AB-test"""
    def __init__(self):
        self.datasets = {}
        self.power = {}
        self.campaigns = {}

    def plot_distributions(self, save_path: str=None) -> None:
        """Generate distributions and save plot on given path."""
        bins = np.linspace(-10, 10, 100)
        plt.hist(self.datasets['A'], bins, alpha=0.5, label='control')
        plt.hist(self.datasets['B'], bins, alpha=0.5, label='treatment')
        plt.legend(loc='upper right')

        if save_path is not None:
            plt.savefig(save_path)
        else:
            plt.show()

    def load_dataset(self, path: str, type: str='discrete', output: str=None,
                     split_by: str=None, confound: str=None) -> None:
        """Load dataset for AB-testing"""
        self.datasets['type'] = type
        df = read_file(path)

        if confound is None:
            self.datasets['A'] = df.loc[df[split_by] == 'control', output]
            self.datasets['B'] = df.loc[df[split_by] == 'treatment', output]

    def generate_datasets(self, n_samples: int=20000, dist1: str='normal', dist1_params: tuple=(0, 1),
                          dist2: str='normal', dist2_params: tuple=(2, 1.1),
                          to_save: bool=False, save_path: str='./data/test_dataset.csv') -> None:
        """Generate two datasets with given parameters for analysis."""
        n_samples_each = n_samples // 2
        a_response = [*generate_distribution(dist1, dist1_params, n_samples_each)]
        b_response = [*generate_distribution(dist2, dist2_params, n_samples_each)]

        campaign_id = randint(1, 50)
        dataset = pd.DataFrame(columns=['user_id', 'campaign_id', 'group', 'response'])
        dataset['user_id'] = range(n_samples)
        dataset['group'] = ['control'] * n_samples_each + ['treatment'] * n_samples_each
        dataset['response'] = a_response + b_response
        dataset['timestamp'] = range(n_samples)
        dataset = dataset.sample(frac=1)

        self.datasets = {
            'A': { 'data': a_response, 'timestamp': dataset['timestamp'][:n_samples_each].tolist() },
            'B': { 'data': b_response, 'timestamp': dataset['timestamp'][n_samples_each:].tolist() }
        }
        self.campaigns[campaign_id] = dataset

        if to_save:
            dataset.to_csv(save_path, index=False)

    def power_analysis(self, power: float=0.8, alpha: float=0.05, ratio: float=1.0,
                       effect_size: float=None, n_samples: float=None) -> dict:
        """Perform power analysis and return computed parameter which was initialised as None."""
        self.alpha = alpha
        unknown_arg = 'n_samples'
        for arg in [*locals().keys()][1:]:
            if eval(arg) is None:
                unknown_arg = arg

        result = sms.TTestIndPower().solve_power(
            effect_size=effect_size,
            power=power,
            nobs1=n_samples,
            alpha=alpha,
            ratio=ratio
        )
        if unknown_arg == 'n_samples':
            result = int(math.ceil(result))
            self.min_sample_size = result
        else:
            result = round(result, 3)
            self.min_sample_size = n_samples

        output = {
            'power': power,
            'alpha': alpha,
            'effect_size': effect_size,
            'n_samples': n_samples,
            'ratio': ratio,
            'beta': None
        }
        output[unknown_arg] = result
        output['beta'] = round(1 - output['power'], 3)

        self.power = output
        return output

    def run_simulation(self, output_path: str='./data/sim_output.xlsx') -> None:
        """Run simulations and save results into file."""
        output = pd.DataFrame(columns=['iteration', 'control', 'treatment', 'statistic', 'pvalue', 'inference'])
        A = pd.DataFrame(self.datasets['A'])
        A['group'] = 'control'
        B = pd.DataFrame(self.datasets['B'])
        B['group'] = 'treatment'
        dataset = pd.concat([A, B])
        dataset.sort_values(by='timestamp', inplace=True)

        for row_index in range(1, dataset.shape[0]):
            series = {'iteration': row_index, 'control': 0, 'treatment': 0, 'statistic': '',
                      'pvalue': '', 'inference': ''}
            data = dataset.iloc[:row_index]
            
            groups = Counter(data['group'])
            series = {**series, **groups}

            if (groups['control'] < 20) or (groups['treatment'] < 20): # условие теста Манна-Уитни
                continue
            elif (groups['control'] > (self.min_sample_size) + 200) \
                    and (groups['treatment'] > (self.min_sample_size + 200)):
                break
            else:
                a = data.loc[data['group'] == 'control', 'data'].tolist()
                b = data.loc[data['group'] == 'treatment', 'data'].tolist()
                
                series['statistic'], series['pvalue'] = mannwhitneyu(a, b, alternative='two-sided')
                series['pvalue'] = round(series['pvalue'], 4)
                series['inference'] = 'Same' if series['pvalue'] > self.alpha else 'Different'
            output = output.append(pd.Series(series), ignore_index=True)
            
        output.to_excel(output_path, index=False)


if __name__ == '__main__':
    # Example scenario to use
    m = ABtest()
    m.generate_datasets(n_samples=1000, dist1='normal', dist1_params=(-2, 1), dist2='normal', dist2_params=(4, 1.1))
    m.load_dataset('./data/test_dataset.csv', type='continuous', output='response', split_by='group', confound=None)
    m.plot_distributions(save_path='./output/AB_dists.png')
    m.power_analysis(n_samples=500, effect_size=0.08, power=None)
    m.run_simulation()

