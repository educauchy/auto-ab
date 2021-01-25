import numpy as np
from random import randint
import pandas as pd
from numpy.random import normal, binomial
import statsmodels.stats.api as sms
import math
from collections import Counter
from scipy.stats import mannwhitneyu
import matplotlib.pyplot as plt


def generate_distribution(dist, params, n_samples):
    """Return distribution with given parameters and number of samples."""
    if dist == 'normal':
        return normal(*params, n_samples)
    elif dist == 'binomial':
        return binomial(*params, n_samples)


class ABtest:
    """Perform AB-test"""
    def __init__(self):
        self.power = {}

    def plot_distibutions(self, a, b, plot_path):
        """Generate distributions and save plot on given path."""
        bins = np.linspace(-10, 10, 100)
        plt.hist(a, bins, alpha=0.5, label='control')
        plt.hist(b, bins, alpha=0.5, label='treatment')
        plt.legend(loc='upper right')
        plt.savefig(plot_path)

    def generate_datasets(self, n_samples:int=20000, dist1:str='normal', dist1_params:tuple=(0, 1),
                          dist2:str='normal', dist2_params:tuple=(2, 1.1),
                          to_plot:bool=False, plot_path:str='./output/distributions.png',
                          to_save:bool=True, save_path:str='./data/test_dataset.csv'):
        """Generate two datasets with given parameters for analysis."""
        n_samples_each = n_samples // 2
        a_response = [*generate_distribution(dist1, dist1_params, n_samples_each)]
        b_response = [*generate_distribution(dist2, dist2_params, n_samples_each)]

        self.dataset = pd.DataFrame(columns=['user_id', 'campaign_id', 'group', 'response'])
        self.dataset['user_id'] = range(n_samples)
        self.dataset['campaign_id'] = [randint(1, 50)] * n_samples
        self.dataset['group'] = ['control'] * n_samples_each + ['treatment'] * n_samples_each
        self.dataset['response'] = a_response + b_response
        
        # перемешиваем датафрейм для элемента случайности
        self.dataset = self.dataset.sample(frac=1)

        if to_save:
            self.dataset.to_csv(save_path, index=False)

        if to_plot:
            self.plot_distibutions(a_response, b_response, plot_path)

    def power_analysis(self, power:float=0.8, alpha:float=0.05, effect_size=None, n_samples=None):
        """Perform power analysis and return computed parameter which was initialised as None."""
        self.alpha = alpha
        for arg in [*locals().keys()][1:]:
            if eval(arg) is None:
                unknown_arg = arg

        result = sms.TTestIndPower().solve_power(
            effect_size=effect_size,
            power=power,
            nobs1=n_samples,
            alpha=alpha,
            ratio=1
        )
        if unknown_arg == 'n_samples':
            result = int(math.ceil(result))
            self.min_sample_size = result
        else:
            result = round(result, 3)

        output = {
            'power': power,
            'alpha': alpha,
            'effect_size': effect_size,
            'n_samples': n_samples
        }
        output[unknown_arg] = result

        return output

    def run_simulation(self, output_path='./data/sim_output.xlsx'):
        """Run simulations and save results into file."""
        output = pd.DataFrame(columns=['iteration', 'control', 'treatment', 'statistic', 'pvalue', 'inference'])
        for row_index in range(1, self.dataset.shape[0]):
            series = {'iteration': row_index, 'control': 0, 'treatment': 0, 'statistic': '',
                      'pvalue': '', 'inference': ''}
            data = self.dataset.iloc[:row_index]
            
            groups = Counter(data['group'])
            series = {**series, **groups}

            if (groups['control'] < 20) or (groups['treatment'] < 20): # условие теста Манна-Уитни
                continue
            elif (groups['control'] > (self.min_sample_size) + 200) \
                    and (groups['treatment'] > (self.min_sample_size + 200)):
                break
            else:
                a = data.loc[data['group'] == 'control', 'response'].tolist()
                b = data.loc[data['group'] == 'treatment', 'response'].tolist()
                
                series['statistic'], series['pvalue'] = mannwhitneyu(a, b, alternative='two-sided')
                series['pvalue'] = round(series['pvalue'], 4)
                series['inference'] = 'Same' if series['pvalue'] > self.alpha else 'Different'
            output = output.append(pd.Series(series), ignore_index=True)
            
        output.to_excel(output_path, index=False)


if __name__ == '__main__':
    # Example scenario to use
    m = ABtest()
    m.generate_datasets(n_samples=10, to_save=False, to_plot=False,
                        dist1='normal', dist1_params=(-1.5, 1), dist2='normal', dist2_params=(2, 1))
    print(m.power_analysis(effect_size=0.08))
    print(m.power_analysis(n_samples=2000, effect_size=0.08, power=None))
    m.run_simulation()