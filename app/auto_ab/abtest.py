import copy
import warnings

import numpy as np
import pandas as pd
import os
import sys
import yaml
from scipy.stats import mannwhitneyu, ttest_ind, shapiro, mode, t
from typing import Dict, List, Any, Union, Optional, Callable, Tuple
from tqdm.auto import tqdm
from splitter import Splitter
from graphics import Graphics
from variance_reduction import VarianceReduction
from hyperopt import hp, fmin, tpe, Trials, space_eval


metric_name_typing = Union[str, Callable[[np.array], Union[int, float]]]

class ABTest:
    """Perform AB-test"""
    def __init__(self, config: Dict[Any, Any] = None,
                 startup_config: bool = False) -> None:
        if config is not None:
            self.startup_config = startup_config
            self.config: Dict[Any, Any] = {}
            self.config_load(config)
        else:
            raise Exception('You must pass config file')

    def __str__(self):
        return f"ABTest(alpha={self.config['alpha']}, " \
               f"beta={self.config['beta']}, " \
               f"alternative='{self.config['alternative']}')"

    @property
    def alpha(self) -> float:
        return self.config['alpha']

    @alpha.setter
    def alpha(self, value: float) -> None:
        if 0 <= value <= 1:
            self.config['alpha'] = value
        else:
            raise Exception('Alpha must be inside interval [0, 1]. Your input: {}.'.format(value))

    @property
    def beta(self) -> float:
        return self.config['beta']

    @beta.setter
    def beta(self, value: float) -> None:
        if 0 <= value <= 1:
            self.config['beta'] = value
        else:
            raise Exception('Beta must be inside interval [0, 1]. Your input: {}.'.format(value))

    @property
    def split_ratios(self) -> Tuple[float, float]:
        return self.config['split_ratios']

    @split_ratios.setter
    def split_ratios(self, value: Tuple[float, float]) -> None:
        if isinstance(value, tuple) and len(value) == 2 and sum(value) == 1:
            self.config['split_ratios'] = value
        else:
            raise Exception('Split ratios must be a tuple with two shares which has a sum of 1. Your input: {}.'.format(value))

    @property
    def alternative(self) -> str:
        return self.config['alternative']

    @alternative.setter
    def alternative(self, value: str) -> None:
        if value in ['less', 'greater', 'two-sided']:
            self.config['alternative'] = value
        else:
            raise Exception("Alternative must be either 'less', 'greater', or 'two-sided'. Your input: '{}'.".format(value))

    @property
    def metric_type(self) -> str:
        return self.config['metric_type']

    @metric_type.setter
    def metric_type(self, value: str) -> None:
        if value in ['solid', 'ratio']:
            self.config['metric_type'] = value
        else:
            raise Exception("Metric type must be either 'solid' or 'ratio'. Your input: '{}'.".format(value))

    @property
    def metric_name(self) -> metric_name_typing:
        return self.config['metric_name']

    @metric_name.setter
    def metric_name(self, value: metric_name_typing) -> None:
        if value in ['mean', 'median'] or callable(value):
            self.config['metric_name'] = value
        else:
            raise Exception("Metric name must be either 'mean' or 'median'. Your input: '{}'.".format(value))

    @property
    def target(self) -> str:
        return self.config['target']

    @target.setter
    def target(self, value: str) -> None:
        if value in self.dataset.columns:
            self.config['target'] = value
        else:
            raise Exception('Target column name must be presented in dataset. Your input: {}.'.format(value))

    @property
    def denominator(self) -> str:
        return self.config['denominator']

    @denominator.setter
    def denominator(self, value: str) -> None:
        if value in self.dataset.columns:
            self.config['denominator'] = value
        else:
            raise Exception('Denominator column name must be presented in dataset. Your input: {}.'.format(value))

    @property
    def numerator(self) -> str:
        return self.config['numerator']

    @numerator.setter
    def numerator(self, value: str) -> None:
        if value in self.dataset.columns:
            self.config['numerator'] = value
        else:
            raise Exception('Numerator column name must be presented in dataset. Your input: {}.'.format(value))

    @property
    def group_col(self) -> str:
        return self.config['group_col']

    @group_col.setter
    def group_col(self, value: str) -> None:
        if value in self.dataset.columns:
            self.config['group_col'] = value
        else:
            raise Exception('Group column name must be presented in dataset. Your input: {}.'.format(value))

    def config_load(self, config: Dict[Any, Any]) -> None:
        if self.startup_config:
            self.config['alpha']          = config['hypothesis']['alpha']
            self.config['beta']           = config['hypothesis']['beta']
            self.config['alternative']    = config['hypothesis']['alternative']
            self.config['n_buckets']      = config['hypothesis']['n_buckets']
            self.config['n_boot_samples'] = config['hypothesis']['n_boot_samples']
            self.config['split_ratios']   = config['hypothesis']['split_ratios']
            self.config['metric_type']    = config['metric']['type']
            self.config['metric_name']    = config['metric']['name']

            if self.config['metric_name'] == 'custom':
                self.metric = lambda x: np.mean(x)

            if config['data']['path'] != '':
                df: pd.DataFrame = self.load_dataset(config['data']['path'])
                n_rows = df.shape[0] + 1 if config['data']['n_rows'] == -1 else config['data']['n_rows']
                df = df.iloc[:n_rows]
                self.config['dataset'] = df.to_dict()
                self.dataset = df

            self.config['target']         = config['data']['target']
            self.config['predictors']     = config['data']['predictors']
            self.config['numerator']      = config['data']['numerator']
            self.config['denominator']    = config['data']['denominator']
            self.config['covariate']      = config['data']['covariate']
            self.config['group_col']      = config['data']['group_col']
            self.config['id_col']         = config['data']['id_col']
            self.config['is_grouped']     = config['data']['is_grouped']
            self.config['target_prev']    = config['data']['target_prev']
            self.config['predictors_prev']     = config['data']['predictors_prev']

            self.config['control']        = self.dataset.loc[self.dataset[self.config['group_col']] == 'A', \
                                                                 self.config['target']].to_numpy()
            self.config['treatment']      = self.dataset.loc[self.dataset[self.config['group_col']] == 'B', \
                                                                 self.config['target']].to_numpy()

        else:
            self.config['alpha']          = config['alpha']
            self.config['beta']           = config['beta']
            self.config['alternative']    = config['alternative']
            self.config['n_buckets']      = config['n_buckets']
            self.config['n_boot_samples'] = config['n_boot_samples']
            self.config['split_ratios']   = config['split_ratios']
            self.config['metric_type']    = config['metric_type']
            self.config['metric_name']    = config['metric_name']

            if self.config['metric_name'] == 'custom':
                self.metric = lambda x: np.mean(x)

            if config['dataset'] != '':
                self.dataset: pd.DataFrame = pd.DataFrame.from_dict(config['dataset'])
                self.config['dataset'] = config['dataset']

            self.config['target']         = config['target']
            self.config['predictors']     = config['predictors']
            self.config['numerator']      = config['numerator']
            self.config['denominator']    = config['denominator']
            self.config['covariate']      = config['covariate']
            self.config['group_col']      = config['group_col']
            self.config['id_col']         = config['id_col']
            self.config['is_grouped']     = config['is_grouped']
            self.config['target_prev']    = config['target_prev']
            self.config['predictors_prev']     = config['predictors_prev']

            self.config['control']        = config['control']
            self.config['treatment']      = config['treatment']

        # self.splitter: Splitter = None
        # self.split_rates: List[float] = None
        # self.increment_list: List[float] = None
        # self.increment_extra: Dict[str, float] = None

    def _add_increment(self, X: Union[pd.DataFrame, np.array] = None,
                       inc_value: Union[float, int] = None) -> np.array:
        """
        Add constant increment to a list
        :param X: Numpy array to modify
        :param inc_value: Constant addendum to each value
        :returns: Modified X array
        """
        if self.config['metric_type'] == 'solid':
            return X + inc_value
        elif self.config['metric_type'] == 'ratio':
            X.loc[:, 'inced'] = X[self.config['numerator']] + inc_value
            X.loc[:, 'diff'] = X[self.config['denominator']] - X[self.config['numerator']]
            X.loc[:, 'rand_inc'] = np.random.randint(0, X['diff'] + 1, X.shape[0])
            X.loc[:, 'numerator_new'] = X[self.config['numerator']] + X['rand_inc']

            X[self.config['numerator']] = np.where(X['inced'] < X[self.config['denominator']], X['inced'], X['numerator_new'])
            return X[[self.config['numerator'], self.config['denominator']]]

    def _split_data(self, split_rate: float) -> None:
        """
        Add 'group' column
        :param split_rate: Split rate of control/treatment
        :return: None
        """
        split_rate: float = self.config['split_rate'] if split_rate is None else split_rate
        self.dataset = self.config['splitter'].fit(self.dataset,
                                                    self.config['target'],
                                                    self.config['numerator'],
                                                    self.config['denominator'],
                                                    split_rate)

    def _read_file(self, path: str) -> pd.DataFrame:
        """
        Read file and return pandas dataframe
        :param path: Path to file
        :returns: Pandas DataFrame
        """
        _, file_ext = os.path.splitext(path)
        if file_ext == '.csv':
            return pd.read_csv(path, encoding='utf8')
        elif file_ext == '.xls' or file_ext == '.xlsx':
            return pd.read_excel(path, encoding='utf8')

    def _manual_ttest(self, A_mean: float, A_var: float, A_size: int, B_mean: float, B_var: float, B_size: int) -> int:
        t_stat_empirical = (A_mean - B_mean) / (A_var / A_size + B_var / B_size) ** (1/2)
        df = A_size + B_size - 2

        test_result: int = 0
        if self.config['alternative'] == 'two-sided':
            lcv, rcv = t.ppf(self.config['alpha'] / 2, df), t.ppf(1.0 - self.config['alpha'] / 2, df)
            if not (lcv < t_stat_empirical < rcv):
                test_result = 1
        elif self.config['alternative'] == 'left':
            lcv = t.ppf(self.config['alpha'], df)
            if t_stat_empirical < lcv:
                test_result = 1
        elif self.config['alternative'] == 'right':
            rcv = t.ppf(1 - self.config['alpha'], df)
            if t_stat_empirical > rcv:
                test_result = 1

        return test_result

    def _linearize(self):
            X = self.dataset.loc[self.dataset[self.config['group_col']] == 'A']
            K = round(sum(X[self.config['numerator']]) / sum(X[self.config['denominator']]), 4)

            self.dataset.loc[:, f"{self.config['numerator']}_{self.config['denominator']}"] = \
                        self.dataset[self.config['numerator']] - K * self.dataset[self.config['denominator']]
            self.target = f"{self.config['numerator']}_{self.config['denominator']}"

    def _delta_params(self, X: pd.DataFrame) -> Tuple[float, float]:
        """
        Calculated expectation and variance for ratio metric using delta approximation
        :param X: Pandas DataFrame of particular group (A, B, etc)
        :return: Tuple with mean and variance of ratio
        """
        num = X[self.config['numerator']]
        den = X[self.config['denominator']]
        num_mean, den_mean = num.mean(), den.mean()
        num_var, den_var = num.var(), den.var()
        cov = X[[self.config['numerator'], self.config['denominator']]].cov().iloc[0, 1]
        n = len(num)

        bias_correction = (den_mean / num_mean ** 3) * (num_var / n) - cov / (n * num_mean ** 2)
        mean = den_mean / num_mean - 1 + bias_correction
        var = den_var / num_mean ** 2 - 2 * (den_mean / num_mean ** 3) * cov + (den_mean ** 2 / num_mean ** 4) * num_var

        return (mean, var)

    def _taylor_params(self, X: pd.DataFrame) -> Tuple[float, float]:
        """
        Calculated expectation and variance for ratio metric using Taylor expansion approximation
        :param X: Pandas DataFrame of particular group (A, B, etc)
        :return: Tuple with mean and variance of ratio
        """
        num = X[self.config['numerator']]
        den = X[self.config['denominator']]
        mean = num.mean() / den.mean() - X[[self.config['numerator'], self.config['denominator']]].cov()[0, 1] \
               / (den.mean() ** 2) + den.var() * num.mean() / (den.mean() ** 3)
        var = (num.mean() ** 2) / (den.mean() ** 2) * (num.var() / (num.mean() ** 2) - \
                2 * X[[self.config['numerator'], self.config['denominator']]].cov()[0, 1]) \
                / (num.mean() * den.mean() + den.var() / (den.mean() ** 2))

        return (mean, var)

    def set_increment(self, inc_var: List[float] = None, extra_params: Dict[str, float] = None) -> None:
        self.config['increment_list']  = inc_var
        self.config['increment_extra'] = extra_params

    def use_dataset(self, X: pd.DataFrame) -> None:
        """
        Put dataset for analysis
        :param X: Pandas DataFrame for analysis
        """
        self.dataset = X
        self.config['dataset'] = X.to_dict()

    def load_dataset(self, path: str = '') -> pd.DataFrame:
        """
        Load dataset for analysis
        :param path: Path to the dataset for analysis
        """
        return self._read_file(path)

    def ratio_bootstrap(self, X: pd.DataFrame = None, Y: pd.DataFrame = None) -> int:
        if X is None and Y is None:
            X = self.dataset[self.dataset[self.config['group_col']] == 'A']
            Y = self.dataset[self.dataset[self.config['group_col']] == 'B']

        a_metric_total = sum(X[self.config['numerator']]) / sum(X[self.config['denominator']])
        b_metric_total = sum(Y[self.config['numerator']]) / sum(Y[self.config['denominator']])
        origin_mean = b_metric_total - a_metric_total
        boot_diffs = []
        boot_a_metric = []
        boot_b_metric = []

        for _ in tqdm(range(self.config['n_boot_samples'])):
            a_boot = X[X[self.config['id_col']].isin(X[self.config['id_col']].sample(X[self.config['id_col']].nunique(), replace=True))]
            b_boot = Y[Y[self.config['id_col']].isin(Y[self.config['id_col']].sample(Y[self.config['id_col']].nunique(), replace=True))]
            a_boot_metric = sum(a_boot[self.config['numerator']]) / sum(a_boot[self.config['denominator']])
            b_boot_metric = sum(b_boot[self.config['numerator']]) / sum(b_boot[self.config['denominator']])
            boot_a_metric.append(a_boot_metric)
            boot_b_metric.append(b_boot_metric)
            boot_diffs.append(b_boot_metric - a_boot_metric)

        # correction
        boot_mean = np.mean(boot_diffs)
        delta = abs(origin_mean - boot_mean)
        boot_diffs = [boot_diff + delta for boot_diff in boot_diffs]
        delta_a = abs(a_metric_total - np.mean(boot_a_metric))
        delta_b = abs(b_metric_total - np.mean(boot_b_metric))
        boot_a_metric = [boot_a_diff + delta_a for boot_a_diff in boot_a_metric]
        boot_b_metric = [boot_b_diff + delta_b for boot_b_diff in boot_b_metric]

        pd_metric_diffs = pd.DataFrame(boot_diffs)

        left_quant  = self.config['alpha'] / 2
        right_quant = 1 - self.config['alpha'] / 2
        ci = pd_metric_diffs.quantile([left_quant, right_quant])
        ci_left, ci_right = float(ci.iloc[0]), float(ci.iloc[1])

        test_result: int = 0 # 0 - cannot reject H0, 1 - reject H0
        if ci_left > 0 or ci_right < 0: # left border of ci > 0 or right border of ci < 0
            test_result = 1

        return test_result

    def ratio_taylor(self) -> int:
        """
        Calculate expectation and variance of ratio for each group
        and then use t-test for hypothesis testing
        Source: http://www.stat.cmu.edu/~hseltman/files/ratio.pdf
        :return: Hypothesis test result: 0 - cannot reject H0, 1 - reject H0
        """
        X = self.dataset[self.dataset[self.config['group_col']] == 'A']
        Y = self.dataset[self.dataset[self.config['group_col']] == 'B']

        A_mean, A_var = self._taylor_params(X)
        B_mean, B_var = self._taylor_params(Y)
        test_result: int = self._manual_ttest(A_mean, A_var, X.shape[0], B_mean, B_var, Y.shape[0])

        return test_result

    def delta_method(self) -> int:
        """
        Delta method with bias correction for ratios
        Source: https://arxiv.org/pdf/1803.06336.pdf
        :return: Hypothesis test result: 0 - cannot reject H0, 1 - reject H0
        """
        X = self.dataset[self.dataset[self.config['group_col']] == 'A']
        Y = self.dataset[self.dataset[self.config['group_col']] == 'B']

        A_mean, A_var = self._delta_params(X)
        B_mean, B_var = self._delta_params(Y)
        test_result: int = self._manual_ttest(A_mean, A_var, X.shape[0], B_mean, B_var, Y.shape[0])

        return test_result

    def linearization(self) -> None:
        """
        Important: there is an assumption that all data is already grouped by user
        s.t. numerator for user = sum of numerators for user for different time periods
        and denominator for user = sum of denominators for user for different time periods
        Source: https://research.yandex.com/publications/148
        :return: None
        """
        if not self.config['is_grouped']:
            not_ratio_columns = self.dataset.columns[~self.dataset.columns.isin([self.config['numerator'], self.config['denominator']])].tolist()
            df_grouped = self.dataset.groupby(by=not_ratio_columns, as_index=False).agg({
                self.config['numerator']: 'sum',
                self.config['denominator']: 'sum'
            })
            self.dataset = df_grouped
        self._linearize()

    def test_hypothesis(self) -> Tuple[int, float, float]:
        """
        Perform Welch's t-test / Mann-Whitney test for means/medians
        :return: Tuple: (test result: 0 - cannot reject H0, 1 - reject H0,
                        statistics,
                        p-value)
        """
        X = self.config['control']
        Y = self.config['treatment']

        test_result: int = 0
        pvalue: float = 1.0
        stat: float = 0.0
        if self.config['metric_name'] == 'mean':
            normality_passed = shapiro(X)[1] >= self.config['alpha'] and shapiro(Y)[1] >= self.config['alpha']
            if not normality_passed:
                warnings.warn('One or both distributions are not normally distributed')
            stat, pvalue = ttest_ind(X, Y, equal_var=False, alternative=self.config['alternative'])
        elif self.config['metric_name'] == 'median':
            stat, pvalue = mannwhitneyu(X, Y, alternative=self.config['alternative'])
        if pvalue <= self.config['alpha']:
            test_result = 1

        return (test_result, stat, pvalue)

    def test_hypothesis_buckets(self) -> int:
        """
        Perform buckets hypothesis testing
        :return: Test result: 1 - significant different, 0 - insignificant difference
        """
        X = self.config['control']
        Y = self.config['treatment']

        np.random.shuffle(X)
        np.random.shuffle(Y)
        X_new = np.array([ self.metric(x) for x in np.array_split(X, self.config['n_buckets']) ])
        Y_new = np.array([ self.metric(y) for y in np.array_split(Y, self.config['n_buckets']) ])

        test_result: int = 0
        if shapiro(X_new)[1] >= self.config['alpha'] and shapiro(Y_new)[1] >= self.config['alpha']:
            _, pvalue = ttest_ind(X_new, Y_new, equal_var=False, alternative=self.config['alternative'])
            if pvalue <= self.config['alpha']:
                test_result = 1
        else:
            def metric(X: np.array):
                modes, _ = mode(X)
                return sum(modes) / len(modes)
            test_result = self.test_hypothesis_boot_confint()

        return test_result

    def test_hypothesis_strat_confint(self, strata_col: str = '',
                                    weights: Dict[str, float] = None) -> int:
        """
        Perform stratification with confidence interval
        :return: Test result: 1 - significant different, 0 - insignificant difference
        """
        metric_diffs: List[float] = []
        X = self.dataset.loc[self.dataset[self.config['group_col']] == 'A']
        Y = self.dataset.loc[self.dataset[self.config['group_col']] == 'B']
        for _ in tqdm(range(self.config['n_boot_samples'])):
            x_strata_metric = 0
            y_strata_metric = 0
            for strat in weights.keys():
                X_strata = X.loc[X[strata_col] == strat, self.config['target']]
                Y_strata = Y.loc[Y[strata_col] == strat, self.config['target']]
                x_strata_metric += (self.metric(np.random.choice(X_strata, size=X_strata.shape[0] // 2, replace=False)) * weights[strat])
                y_strata_metric += (self.metric(np.random.choice(Y_strata, size=Y_strata.shape[0] // 2, replace=False)) * weights[strat])
            metric_diffs.append(self.metric(x_strata_metric) - self.metric(y_strata_metric))
        pd_metric_diffs = pd.DataFrame(metric_diffs)

        left_quant = self.config['alpha'] / 2
        right_quant = 1 - self.config['alpha'] / 2
        ci = pd_metric_diffs.quantile([left_quant, right_quant])
        ci_left, ci_right = float(ci.iloc[0]), float(ci.iloc[1])

        test_result: int = 0 # 0 - cannot reject H0, 1 - reject H0
        if ci_left > 0 or ci_right < 0: # left border of ci > 0 or right border of ci < 0
            test_result = 1

        return test_result

    def test_hypothesis_boot_est(self) -> float:
        """
        Perform bootstrap confidence interval with
        :returns: Type I error rate
        """
        X = self.config['control']
        Y = self.config['treatment']

        metric_diffs: List[float] = []
        for _ in tqdm(range(self.config['n_boot_samples'])):
            x_boot = np.random.choice(X, size=X.shape[0], replace=True)
            y_boot = np.random.choice(Y, size=Y.shape[0], replace=True)
            metric_diffs.append( self.metric(x_boot) - self.metric(y_boot) )
        pd_metric_diffs = pd.DataFrame(metric_diffs)

        left_quant = self.config['alpha'] / 2
        right_quant = 1 - self.config['alpha'] / 2
        ci = pd_metric_diffs.quantile([left_quant, right_quant])
        ci_left, ci_right = float(ci.iloc[0]), float(ci.iloc[1])

        criticals = [0, 0]
        for boot in pd_metric_diffs:
            if boot < 0 and boot < ci_left:
                criticals[0] += 1
            elif boot > 0 and boot > ci_right:
                criticals[1] += 1
        false_positive = min(criticals) / pd_metric_diffs.shape[0]

        return false_positive

    def test_hypothesis_boot_confint(self) -> int:
        """
        Perform bootstrap confidence interval
        :returns: Ratio of rejected H0 hypotheses to number of all tests
        """
        X = self.config['control']
        Y = self.config['treatment']

        metric_diffs: List[float] = []
        for _ in tqdm(range(self.config['n_boot_samples'])):
            x_boot = np.random.choice(X, size=X.shape[0], replace=True)
            y_boot = np.random.choice(Y, size=Y.shape[0], replace=True)
            metric_diffs.append( self.metric(x_boot) - self.metric(y_boot) )
        pd_metric_diffs = pd.DataFrame(metric_diffs)

        left_quant = self.config['alpha'] / 2
        right_quant = 1 - self.config['alpha'] / 2
        ci = pd_metric_diffs.quantile([left_quant, right_quant])
        ci_left, ci_right = float(ci.iloc[0]), float(ci.iloc[1])

        test_result: int = 0 # 0 - cannot reject H0, 1 - reject H0
        if ci_left > 0 or ci_right < 0: # left border of ci > 0 or right border of ci < 0
            test_result = 1

        return test_result

    def test_boot_hypothesis(self) -> float:
        """
        Perform T-test for independent samples with unequal number of observations and variance
        :returns: Ratio of rejected H0 hypotheses to number of all tests
        """
        X = self.config['control']
        Y = self.config['treatment']

        T: int = 0
        for _ in range(self.config['n_boot_samples']):
            x_boot = np.random.choice(X, size=X.shape[0], replace=True)
            y_boot = np.random.choice(Y, size=Y.shape[0], replace=True)

            T_boot = (np.mean(x_boot) - np.mean(y_boot)) / (np.var(x_boot) / x_boot.shape[0] + np.var(y_boot) / y_boot.shape[0])
            test_res = ttest_ind(x_boot, y_boot, equal_var=False, alternative=self.config['alternative'])

            if T_boot >= test_res[1]:
                T += 1

        pvalue = T / self.config['n_boot_samples']

        return pvalue

    def sample_size(self, std: float = None, effect_size: float = None,
                    split_ratios: Tuple[float, float] = None) -> Tuple[int, int]:
        """
        Calculation of sample size for each test group
        :param std: Standard deviation of a test metric
        :param effect_size: Lift in metric
        :param group_shares: Shares of A and B groups
        :return: Number of observations needed in each group
        """
        control_share, treatment_share = split_ratios if split_ratios is not None else self.config['split_ratios']
        if treatment_share == 0.5:
            alpha: float = (1 - self.config['alpha'] / 2) if self.config['alternative'] == 'two-sided' else (1 - self.config['alpha'])
            n_samples: int = round(2 * (t.ppf(alpha) + t.ppf(1 - self.config['beta'])) * std ** 2 / (effect_size ** 2), 0) + 1
            return (n_samples, n_samples)
        else:
            alpha: float = (1 - self.config['alpha'] / 2) if self.config['alternative'] == 'two-sided' else (1 - self.config['alpha'])
            n: int = round((((t.ppf(alpha) + t.ppf(1 - self.config['beta'])) * std ** 2 / (effect_size ** 2))) \
                      / (treatment_share * control_share), 0) + 1
            a_samples, b_samples = int(round(n * control_share, 0) + 1), int(round(n * treatment_share, 0) + 1)
            return (a_samples, b_samples)

    def mde(self, std: float = None, n_samples: int = None) -> float:
        """
        Calculate Minimum Detectable Effect using Margin of Error formula
        :param std: Pooled standard deviatioin
        :param n_samples: Number of samples for each group
        :return: MDE, in absolute lift
        """
        alpha: float = (1 - self.config['alpha'] / 2) if self.config['alternative'] == 'two-sided' else (1 - self.config['alpha'])
        mde: float = np.sqrt( 2 * (t.ppf(alpha) + t.ppf(1 - self.config['beta'])) * std / n_samples )
        return mde


    def plot(self) -> None:
        a = self.__get_group('A')
        b = self.__get_group('B')

        if self.config['metric_name'] == 'mean':
            Graphics().plot_mean_experiment(a, b,
                               self.config['alternative'],
                               self.config['metric_name'],
                               self.config['alpha'],
                               self.config['beta'])

    def __get_group(self, group_label: str = 'A'):
        group = self.dataset.loc[self.dataset[self.config['group_col']] == group_label, \
                                            self.config['target']].to_numpy()
        return group

    def cuped(self):
        vr = VarianceReduction()
        result_df = vr.cuped(self.dataset,
                            target=self.config['target'],
                            groups=self.config['group_col'],
                            covariate=self.config['covariate'])

        self.config_new = copy.deepcopy(self.config)

        self.config_new['dataset'] = result_df.to_dict()
        self.dataset = result_df

        self.config_new['control'] = self.__get_group('A')
        self.config_new['treatment'] = self.__get_group('B')

        return ABTest(self.config_new)

    def cupac(self):
        vr = VarianceReduction()
        result_df = vr.cupac(self.dataset,
                               target_prev=self.config['target_prev'],
                               target_now=self.config['target'],
                               factors_prev=self.config['predictors_prev'],
                               factors_now=self.config['predictors'],
                               groups=self.config['group_col'])

        self.config_new = copy.deepcopy(self.config)

        self.config_new['dataset'] = result_df.to_dict()
        self.dataset = result_df

        self.config_new['control'] = self.__get_group('A')
        self.config_new['treatment'] = self.__get_group('B')

        return ABTest(self.config_new)

    def __metric_calc(self, X: Union[List[Any], np.array]):
        if self.config['metric_name'] == 'mean':
            return np.mean(X)
        elif self.config['metric_name'] == 'median':
            return np.median(X)
        elif self.config['metric_name'] == 'custom':
            return self.config['metric'](X)

    def __bucketize(self, X: pd.DataFrame):
        np.random.shuffle(X)
        X_new = np.array([ self.__metric_calc(x) for x in np.array_split(X, self.config['n_buckets']) ])
        return X_new

    def bucketing(self):
        self.config_new = copy.deepcopy(self.config)

        self.config_new['control']   = self.__bucketize(self.config['control'])
        self.config_new['treatment'] = self.__bucketize(self.config['treatment'])

        return ABTest(self.config_new)


if __name__ == '__main__':
    df = pd.read_csv('../../examples/storage/data/ab_data.csv')

    with open("../../config.yaml", "r") as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    ab_obj = ABTest(config, startup_config=True)
    ab_obj = ab_obj.cupac().bucketing()
    print(ab_obj.config)

    # or with CUPED
    # ab_obj = ab_obj.cuped().bucketing()
