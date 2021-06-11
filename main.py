import numpy as np
import pandas as pd
import statsmodels.stats.api as sms
import math, os, sys, yaml
import matplotlib.pyplot as plt
from collections import Counter
from scipy.stats import mannwhitneyu, ttest_1samp, ttest_ind, ttest_ind_from_stats, ttest_rel
from typing import Dict, List, Tuple


try:
    project_dir = os.path.dirname(__file__)
    config_file = os.path.join(project_dir, 'config.yaml')
    with open (config_file, 'r') as file:
        config = yaml.safe_load(file)
except yaml.YAMLError as exc:
    print(exc)
    sys.exit(1)
except Exception as e:
    print('Error reading the config file')
    sys.exit(1)




def generate_distribution(dist_type: str, params: tuple, n_samples: int) -> np.array:
    """Return distribution with given parameters and number of samples."""
    if dist_type == 'normal':
        return np.random.normal(*params, n_samples)
    elif dist_type == 'binomial':
        return np.random.binomial(*params, n_samples)

def read_file(path: str) -> pd.DataFrame:
    """Read file and return pandas dataframe"""
    _, file_ext = os.path.splitext(path)
    if file_ext == '.csv':
        df = pd.read_csv(path, encoding='utf8')
    elif file_ext == '.xls' or file_ext == '.xlsx':
        df = pd.read_excel(path, encoding='utf8')
    return df



class ABTest:
    """Perform AB-test"""
    def __init__(self, alpha: float = 0.05, alternative: str = 'one-sided', n_boot_samples: int = 10000) -> None:
        self.alpha = alpha
        self.alternative = alternative
        self.n_boot_samples = n_boot_samples
        self.datasets = { 'A': {}, 'B': {}, 'type': 'continuous'}
        self.power = {}
        self.campaigns = {}

    def plot_distributions(self, save_path: str=None) -> None:
        """Generate distributions and save plot on given path."""
        bins = np.linspace(-10, 10, 100)
        plt.hist(self.datasets['A']['data'], bins, alpha=0.5, label='control')
        plt.hist(self.datasets['B']['data'], bins, alpha=0.5, label='treatment')
        plt.legend(loc='upper right')

        if save_path is not None:
            plt.savefig(save_path)
        else:
            plt.show()

    def ttest_ind(self, X: np.array, Y: np.array, equal_var: bool = False, use_bootstrap: bool = False) -> tuple:
        T: List[float] = []
        for _ in range(self.n_boot_samples):
            x_boot = np.random.choice(X, size=X.shape[0], replace=True)
            y_boot = np.random.choice(Y, size=Y.shape[0], replace=True)
            test_res = ttest_ind(x_boot, y_boot, equal_var=equal_var, alternative=self.alternative)
            if test_res[1] >= self.alpha:
                T.append(test_res[0])
        boot_alpha = np.sum(T) / self.n_boot_samples
        return boot_alpha

    def load_datasets(self, X: pd.DataFrame, Y: pd.DataFrame) -> None:
        self.datasets['A'] = X
        self.datasets['B'] = Y

    def use_datasets(self, X: np.array, Y: np.array) -> None:
        self.datasets['A'] = X
        self.datasets['B'] = Y

    def load_dataset(self, path: str, type: str='discrete', output: str=None,
                     split_by: str=None, confound: str=None) -> None:
        """Load dataset for AB-testing"""
        self.datasets['type'] = type
        df = read_file(path)

        if 'timestamp' not in df.columns:
            df['timestamp'] = range(df.shape[0])

        if confound is None:
            self.datasets['A']['data'] = df.loc[df[split_by] == 'control', output]
            self.datasets['A']['timestamp'] = df.loc[df[split_by] == 'control', 'timestamp']
            self.datasets['B']['data'] = df.loc[df[split_by] == 'treatment', output]
            self.datasets['B']['timestamp'] = df.loc[df[split_by] == 'treatment', 'timestamp']

    def generate_datasets(self, n_samples: int=20000, dist1: str='normal', dist1_params: tuple=(0, 1),
                          dist2: str='normal', dist2_params: tuple=(2, 1.1),
                          to_save: bool=False, save_path: str='./data/test_dataset.csv') -> None:
        """Generate two datasets with given parameters for analysis."""
        n_samples_each = n_samples // 2
        a_response = [*generate_distribution(dist1, dist1_params, n_samples_each)]
        b_response = [*generate_distribution(dist2, dist2_params, n_samples_each)]

        # campaign_id = randint(1, 50)
        campaign_id = np.random.randint(1, 50, 1)[0]
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
                       effect_size: float=None, n_samples: float=None) -> Dict[str, float]:
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

        # print(dataset.head())
        print(pd.crosstab(dataset['group'], dataset['data']))

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
    m = ABTest(alpha=0.5, alternative='two-sided', n_boot_samples=10000)
    x = np.random.normal(0, 1, 10000)
    y = np.random.normal(0, 1, 20000)
    m.use_datasets(x, y)
    res = m.ttest_ind(x, y, equal_var=False, use_bootstrap=True)
    print(res)


    # m.generate_datasets(n_samples=config['n_samples'], dist1=config['dist1'], dist1_params=config['dist1_params'], \
    #                     dist2=config['dist2'], dist2_params=config['dist2_params'])
    # m.load_dataset('./data/test_dataset.csv', type='continuous', output='response', split_by='group', confound=None)
    # m.plot_distributions(save_path='./output/AB_dists.png')
    # print(m.power_analysis(n_samples=config['power']['n_samples'], effect_size=config['power']['effect_size'], power=None))
    # m.run_simulation()

