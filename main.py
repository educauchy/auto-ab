import numpy as np
from random import sample, randint
import pandas as pd
from numpy.random import normal, binomial
import statsmodels.stats.api as sms
import math
from collections import Counter
from scipy.stats import mannwhitneyu
import matplotlib.pyplot as plt


def generate_distribution(dist, params, n_samples):
    if dist == 'normal':
        return normal(*params, n_samples)
    elif dist == 'binomial':
        return binomial(*params, n_samples)


class ABtest:
    def __init__(self):
        self.power = {}

    def plot_distibutions(self, a, b, plot_path):
        bins = np.linspace(-10, 10, 100)
        plt.hist(a, bins, alpha=0.5, label='control')
        plt.hist(b, bins, alpha=0.5, label='treatment')
        plt.legend(loc='upper right')
        plt.savefig(plot_path)

    def generate_datasets(self, n_samples:int=20000, dist1:str='normal', dist1_params:tuple=(0, 1),
                          dist2:str='normal', dist2_params:tuple=(2, 1.1),
                          to_plot:bool=False, plot_path:str='./output/distributions.png',
                          to_save:bool=True, save_path:str='./data/test_dataset.csv'):
        n_samples_each = n_samples // 2

        self.dataset = pd.DataFrame(columns=['user_id', 'campaign_id', 'group', 'response'])
        self.dataset['user_id'] = range(n_samples)
        self.dataset['campaign_id'] = [randint(1, 50)] * n_samples

        a_response = [*generate_distribution(dist1, dist1_params, n_samples_each)]
        b_response = [*generate_distribution(dist2, dist2_params, n_samples_each)]

        self.dataset['group'] = ['control'] * n_samples_each + ['treatment'] * n_samples_each
        self.dataset['response'] = a_response + b_response
        self.dataset = self.dataset.sample(frac=1)

        if to_save:
            self.dataset.to_csv(save_path, index=False)

        if to_plot:
            self.plot_distibutions(a_response, b_response, plot_path)

    def power_analysis(self, power:float=0.8, alpha:float=0.05, effect_size=None, n_samples=None):
        self.alpha = alpha

        # TODO: переписать функцию, чтобы возвращала словарь со всеми параметрами + подсчитанный новый
        result = sms.TTestIndPower().solve_power(
            effect_size=effect_size,
            power=power,
            nobs1=n_samples,
            alpha=alpha,
            ratio=1
        )
        result = int(math.ceil(result)) if n_samples is None else result
        self.min_sample_size = result if n_samples is None else n_samples
        return result

    def run_simulation(self, output_path='./data/sim_output.xlsx'):
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
    m.generate_datasets(n_samples=10000, to_save=True, to_plot=True,
                        dist1='normal', dist1_params=(-1.5, 1), dist2='normal', dist2_params=(2, 1))
    print(m.power_analysis(effect_size=0.07))
    m.run_simulation()