import warnings

import numpy as np
import pandas as pd
import os
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
    def __init__(self,
                 config: Dict[Any, Any] = None) -> None:
        if config is not None:
            self.config(config)
        else:
            raise Exception('You must pass config file')

    def __str__(self):
        return f"ABTest(alpha={self.__alpha}, beta={self.__beta}, alternative='{self.__alternative}')"

    @property
    def alpha(self) -> float:
        return self.__alpha

    @alpha.setter
    def alpha(self, value: float) -> None:
        if 0 <= value <= 1:
            self.__alpha = value
        else:
            raise Exception('Alpha must be inside interval [0, 1]. Your input: {}.'.format(value))

    @property
    def beta(self) -> float:
        return self.__beta

    @beta.setter
    def beta(self, value: float) -> None:
        if 0 <= value <= 1:
            self.__beta = value
        else:
            raise Exception('Beta must be inside interval [0, 1]. Your input: {}.'.format(value))

    @property
    def split_ratios(self) -> Tuple[float, float]:
        return self.__split_ratios

    @split_ratios.setter
    def split_ratios(self, value: Tuple[float, float]) -> None:
        if isinstance(value, tuple) and len(value) == 2 and sum(value) == 1:
            self.__split_ratios = value
        else:
            raise Exception('Split ratios must be a tuple with two shares which has a sum of 1. Your input: {}.'.format(value))

    @property
    def alternative(self) -> str:
        return self.__alternative

    @alternative.setter
    def alternative(self, value: str) -> None:
        if value in ['less', 'greater', 'two-sided']:
            self.__alternative = value
        else:
            raise Exception("Alternative must be either 'less', 'greater', or 'two-sided'. Your input: '{}'.".format(value))

    @property
    def metric_type(self) -> str:
        return self.__metric_type

    @metric_type.setter
    def metric_type(self, value: str) -> None:
        if value in ['solid', 'ratio']:
            self.__metric_type = value
        else:
            raise Exception("Metric type must be either 'solid' or 'ratio'. Your input: '{}'.".format(value))

    @property
    def metric_name(self) -> metric_name_typing:
        return self.__metric_name

    @metric_name.setter
    def metric_name(self, value: metric_name_typing) -> None:
        if value in ['mean', 'median'] or callable(value):
            self.__metric_name = value
        else:
            raise Exception("Metric name must be either 'mean' or 'median'. Your input: '{}'.".format(value))

    @property
    def target(self) -> str:
        return self.__target

    @target.setter
    def target(self, value: str) -> None:
        if value in self.dataset.columns:
            self.__target = value
        else:
            raise Exception('Target column name must be presented in dataset. Your input: {}.'.format(value))

    @property
    def denominator(self) -> str:
        return self.__denominator

    @denominator.setter
    def denominator(self, value: str) -> None:
        if value in self.dataset.columns:
            self.__denominator = value
        else:
            raise Exception('Denominator column name must be presented in dataset. Your input: {}.'.format(value))

    @property
    def numerator(self) -> str:
        return self.__numerator

    @numerator.setter
    def numerator(self, value: str) -> None:
        if value in self.dataset.columns:
            self.__numerator = value
        else:
            raise Exception('Numerator column name must be presented in dataset. Your input: {}.'.format(value))

    @property
    def group_col(self) -> str:
        return self.__group_col

    @group_col.setter
    def group_col(self, value: str) -> None:
        if value in self.dataset.columns:
            self.__group_col = value
        else:
            raise Exception('Group column name must be presented in dataset. Your input: {}.'.format(value))

    def config(self, config: Dict[Any, Any]) -> None:
        self.alpha          = config['hypothesis']['alpha']
        self.beta           = config['hypothesis']['beta']
        self.alternative    = config['hypothesis']['alternative']
        self.n_buckets      = config['hypothesis']['n_buckets']
        self.metric_type    = config['metric']['metric_type']
        self.metric_name    = config['metric']['metric_name']
        self.split_ratios   = config['data']['split_ratios']

        if config['data']['path'] != '':
            self.dataset: pd.DataFrame = self.load_dataset(config['data']['path'])

        self.target         = config['data']['target']
        self.numerator      = config['data']['numerator']
        self.denominator    = config['data']['denominator']
        self.group_col      = config['data']['group_col']
        self.id_col         = config['data']['id_col']
        self.control        = self.dataset.loc[self.dataset[self.__group_col] == 'A', self.__target].to_numpy()
        self.treatment      = self.dataset.loc[self.dataset[self.__group_col] == 'B', self.__target].to_numpy()

        # self.initial_dataset: pd.DataFrame = None    # for ratio metrics to keep old dataset
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
        if self.__metric_type == 'solid':
            return X + inc_value
        elif self.__metric_type == 'ratio':
            X.loc[:, 'inced'] = X[self.__numerator] + inc_value
            X.loc[:, 'diff'] = X[self.__denominator] - X[self.__numerator]
            X.loc[:, 'rand_inc'] = np.random.randint(0, X['diff'] + 1, X.shape[0])
            X.loc[:, 'numerator_new'] = X[self.__numerator] + X['rand_inc']

            X[self.__numerator] = np.where(X['inced'] < X[self.__denominator], X['inced'], X['numerator_new'])
            return X[[self.__numerator, self.__denominator]]

    def _split_data(self, split_rate: float) -> None:
        """
        Add 'group' column
        :param split_rate: Split rate of control/treatment
        :return: None
        """
        split_rate: float = self.split_rate if split_rate is None else split_rate
        self.dataset = self.splitter.fit(self.dataset, self.__target, self.__numerator, self.__denominator, split_rate)

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
        if self.__alternative == 'two-sided':
            lcv, rcv = t.ppf(self.__alpha / 2, df), t.ppf(1.0 - self.__alpha / 2, df)
            if not (lcv < t_stat_empirical < rcv):
                test_result = 1
        elif self.__alternative == 'left':
            lcv = t.ppf(self.__alpha, df)
            if t_stat_empirical < lcv:
                test_result = 1
        elif self.__alternative == 'right':
            rcv = t.ppf(1 - self.__alpha, df)
            if t_stat_empirical > rcv:
                test_result = 1

        return test_result

    def _linearize(self):
            X = self.dataset.loc[self.dataset[self.__group_col] == 'A']
            K = round(sum(X[self.__numerator]) / sum(X[self.__denominator]), 4)
            self.dataset.loc[:, f'{self.__numerator}_{self.__denominator}'] = \
                        self.dataset[self.__numerator] - K * self.dataset[self.__denominator]
            self.target = f'{self.__numerator}_{self.__denominator}'

    def _delta_params(self) -> Tuple[float, float]:
        """
        Calculated expectation and variance for ratio metric using delta approximation
        :param df: Pandas DataFrame of particular group (A, B, etc)
        :param numerator: Ratio numerator column name
        :param denominator: Ratio denominator column name
        :return: Tuple with mean and variance of ratio
        """
        num = self.dataset[self.__numerator]
        den = self.dataset[self.__denominator]
        num_mean = num.mean()
        num_var = num.var()
        den_mean = den.mean()
        den_var = den.var()
        cov = self.dataset[[self.__numerator, self.__denominator]].cov().iloc[0, 1]
        n = len(num)

        bias_correction = (den_mean / num_mean ** 3) * (num_var / n) - cov / (n * num_mean ** 2)
        mean = den_mean / num_mean - 1 + bias_correction
        var = den_var / num_mean ** 2 - 2 * (den_mean / num_mean ** 3) * cov + (den_mean ** 2 / num_mean ** 4) * num_var

        return (mean, var)

    def _taylor_params(self) -> Tuple[float, float]:
        """
        Calculated expectation and variance for ratio metric using Taylor expansion approximation
        :param df: Pandas DataFrame of particular group (A, B, etc)
        :param numerator: Ratio numerator column name
        :param denominator: Ratio denominator column name
        :return: Tuple with mean and variance of ratio
        """
        num = self.dataset[self.__numerator]
        den = self.dataset[self.__denominator]
        mean = num.mean() / den.mean() - self.dataset[[self.__numerator, self.__denominator]].cov()[0, 1] \
               / (den.mean() ** 2) + den.var() * num.mean() / (den.mean() ** 3)
        var = (num.mean() ** 2) / (den.mean() ** 2) * (num.var() / (num.mean() ** 2) - \
                2 * self.dataset[[self.__numerator, self.___denominator]].cov()[0, 1]) \
                / (num.mean() * den.mean() + den.var() / (den.mean() ** 2))

        return (mean, var)

    def set_increment(self, inc_var: List[float] = None, extra_params: Dict[str, float] = None) -> None:
        self.increment_list = inc_var
        self.increment_extra = extra_params

    def use_dataset(self, X: pd.DataFrame) -> None:
        """
        Put dataset for analysis
        :param X: Pandas DataFrame for analysis
        :param id_col: Id column name
        :param target: Target column name
        :param numerator: Ratio numerator column name
        :param denominator: Ratio denominator column name
        """
        self.dataset = X

    def load_dataset(self, path: str = '') -> None:
        """
        Load dataset for analysis
        :param path: Path to the dataset for analysis
        :param id_col: Id column name
        :param target: Target column name
        :param numerator: Ratio numerator column name
        :param denominator: Ratio denominator column name
        """
        self.dataset = self._read_file(path)

    def ratio_bootstrap(self, X: pd.DataFrame = None, Y: pd.DataFrame = None,
                        n_boot_samples: int = 5000) -> int:
        if X is None and Y is None:
            X = self.dataset[self.dataset[self.__group_col] == 'A']
            Y = self.dataset[self.dataset[self.__group_col] == 'B']

        a_metric_total = sum(X[self.__numerator]) / sum(X[self.__denominator])
        b_metric_total = sum(Y[self.__numerator]) / sum(Y[self.__denominator])
        origin_mean = b_metric_total - a_metric_total
        boot_diffs = []
        boot_a_metric = []
        boot_b_metric = []

        for _ in tqdm(range(n_boot_samples)):
            a_boot = X[X[self.id_col].isin(X[self.id_col].sample(X[self.id_col].nunique(), replace=True))]
            b_boot = Y[Y[self.id_col].isin(Y[self.id_col].sample(Y[self.id_col].nunique(), replace=True))]
            a_boot_metric = sum(a_boot[self.__numerator]) / sum(a_boot[self.__denominator])
            b_boot_metric = sum(b_boot[self.__numerator]) / sum(b_boot[self.__denominator])
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

        left_quant = self.__alpha / 2
        right_quant = 1 - self.__alpha / 2
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
        :param numerator: Ratio numerator column name
        :param denominator: Ratio denominator column name
        :return: Hypothesis test result: 0 - cannot reject H0, 1 - reject H0
        """
        X = self.dataset[self.dataset[self.__group_col] == 'A']
        Y = self.dataset[self.dataset[self.__group_col] == 'B']

        A_mean, A_var = self._taylor_params(X, self.__numerator, self.__denominator)
        B_mean, B_var = self._taylor_params(Y, self.__numerator, self.__denominator)
        test_result: int = self._manual_ttest(A_mean, A_var, X.shape[0], B_mean, B_var, Y.shape[0])

        return test_result

    def delta_method(self) -> int:
        """
        Delta method with bias correction for ratios
        Source: https://arxiv.org/pdf/1803.06336.pdf
        :param X: Group A
        :param Y: Group B
        :return: Hypothesis test result: 0 - cannot reject H0, 1 - reject H0
        """
        X = self.dataset[self.dataset[self.__group_col] == 'A']
        Y = self.dataset[self.dataset[self.__group_col] == 'B']

        A_mean, A_var = self._delta_params(X, self.__numerator, self.__denominator)
        B_mean, B_var = self._delta_params(Y, self.__numerator, self.__denominator)
        test_result: int = self._manual_ttest(A_mean, A_var, X.shape[0], B_mean, B_var, Y.shape[0])

        return test_result

    def linearization(self, is_grouped: bool = False) -> None:
        """
        Important: there is an assumption that all data is already grouped by user
        s.t. numerator for user = sum of numerators for user for different time periods
        and denominator for user = sum of denominators for user for different time periods
        Source: https://research.yandex.com/publications/148
        :param numerator: Ratio numerator column name
        :param denominator: Ratio denominator column name
        :return: None
        """
        if not is_grouped:
            not_ratio_columns = self.dataset.columns[~self.dataset.columns.isin([self.__numerator, self.__denominator])].tolist()
            df_grouped = self.dataset.groupby(by=not_ratio_columns, as_index=False).agg({
                self.__numerator: 'sum',
                self.__denominator: 'sum'
            })
            self.initial_dataset = self.dataset.copy(deep=True)
            self.dataset = df_grouped
        self._linearize(self.__numerator, self.__denominator)

    def test_hypothesis(self, X: np.array = None, Y: np.array = None) -> Tuple[int, float, float]:
        """
        Perform Welch's t-test / Mann-Whitney test for means/medians
        :param X: Group A
        :param Y: Group B
        :return: Tuple: (test result: 0 - cannot reject H0, 1 - reject H0,
                        statistics,
                        p-value)
        """
        if X is None or Y is None:
            X = self.dataset.loc[self.dataset[self.__group_col] == 'A', self.__target].to_numpy()
            Y = self.dataset.loc[self.dataset[self.__group_col] == 'B', self.__target].to_numpy()

        test_result: int = 0
        pvalue: float = self.__alpha + 0.01
        stat: float = 0
        if self.metric_name == 'mean':
            normality_passed = shapiro(X)[1] >= self.__alpha and shapiro(Y)[1] >= self.__alpha
            if not normality_passed:
                warnings.warn('One or both distributions are not normally distributed')
            stat, pvalue = ttest_ind(X, Y, equal_var=False, alternative=self.__alternative)
        elif self.metric_name == 'median':
            stat, pvalue = mannwhitneyu(X, Y, alternative=self.__alternative)
        if pvalue <= self.__alpha:
            test_result = 1

        return (test_result, stat, pvalue)

    def test_hypothesis_buckets(self, X: np.array, Y: np.array,
                                metric: Optional[Callable[[Any], float]] = None,
                                n_buckets: int = 1000) -> int:
        """
        Perform buckets hypothesis testing
        :param X: Null hypothesis distribution
        :param Y: Alternative hypothesis distribution
        :param metric: Custom metric (mean, median, percentile (1, 2, ...), etc
        :param n_buckets: Number of buckets
        :return: Test result: 1 - significant different, 0 - insignificant difference
        """
        if X is None or Y is None:
            X = self.dataset.loc[self.dataset[self.__group_col] == 'A', self.__target].to_numpy()
            Y = self.dataset.loc[self.dataset[self.__group_col] == 'B', self.__target].to_numpy()

        np.random.shuffle(X)
        np.random.shuffle(Y)
        X_new = np.array([ metric(x) for x in np.array_split(X, n_buckets) ])
        Y_new = np.array([ metric(y) for y in np.array_split(Y, n_buckets) ])

        test_result: int = 0
        if shapiro(X_new)[1] >= self.__alpha and shapiro(Y_new)[1] >= self.__alpha:
            _, pvalue = ttest_ind(X_new, Y_new, equal_var=False, alternative=self.__alternative)
            if pvalue <= self.__alpha:
                test_result = 1
        else:
            def metric(X: np.array):
                modes, _ = mode(X)
                return sum(modes) / len(modes)
            test_result = self.test_hypothesis_boot_confint(X_new, Y_new, metric=metric)

        return test_result

    def test_hypothesis_strat_confint(self, metric: Optional[Callable[[Any], float]] = None,
                            strata_col: str = '', n_boot_samples: int = 5000,
                            weights: Dict[str, float] = None) -> int:
        """
        Perform stratification with confidence interval
        :param metric: Custom metric (mean, median, percentile (1, 2, ...), etc
        :param strata_col: Column name of strata column
        :return: Test result: 1 - significant different, 0 - insignificant difference
        """
        metric_diffs: List[float] = []
        X = self.dataset.loc[self.dataset[self.__group_col] == 'A']
        Y = self.dataset.loc[self.dataset[self.__group_col] == 'B']
        for _ in tqdm(range(n_boot_samples)):
            x_strata_metric = 0
            y_strata_metric = 0
            for strat in weights.keys():
                X_strata = X.loc[X[strata_col] == strat, self.__target]
                Y_strata = Y.loc[Y[strata_col] == strat, self.__target]
                x_strata_metric += (metric(np.random.choice(X_strata, size=X_strata.shape[0] // 2, replace=False)) * weights[strat])
                y_strata_metric += (metric(np.random.choice(Y_strata, size=Y_strata.shape[0] // 2, replace=False)) * weights[strat])
            metric_diffs.append(metric(x_strata_metric) - metric(y_strata_metric))
        pd_metric_diffs = pd.DataFrame(metric_diffs)

        left_quant = self.__alpha / 2
        right_quant = 1 - self.__alpha / 2
        ci = pd_metric_diffs.quantile([left_quant, right_quant])
        ci_left, ci_right = float(ci.iloc[0]), float(ci.iloc[1])

        test_result: int = 0 # 0 - cannot reject H0, 1 - reject H0
        if ci_left > 0 or ci_right < 0: # left border of ci > 0 or right border of ci < 0
            test_result = 1

        return test_result

    def test_hypothesis_boot_est(self, X: np.array, Y: np.array,
                        n_boot_samples: int = 5000,
                        metric: Optional[Callable[[Any], float]] = None) -> float:
        """
        Perform bootstrap confidence interval with
        :param X: Null hypothesis distribution
        :param Y: Alternative hypothesis distribution
        :param metric: Custom metric (mean, median, percentile (1, 2, ...), etc
        :returns: Type I error rate
        """
        if X is None or Y is None:
            X = self.dataset.loc[self.dataset[self.__group_col] == 'A', self.__target].to_numpy()
            Y = self.dataset.loc[self.dataset[self.__group_col] == 'B', self.__target].to_numpy()

        metric_diffs: List[float] = []
        for _ in tqdm(range(n_boot_samples)):
            x_boot = np.random.choice(X, size=X.shape[0], replace=True)
            y_boot = np.random.choice(Y, size=Y.shape[0], replace=True)
            metric_diffs.append( metric(x_boot) - metric(y_boot) )
        pd_metric_diffs = pd.DataFrame(metric_diffs)

        left_quant = self.__alpha / 2
        right_quant = 1 - self.__alpha / 2
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

    def test_hypothesis_boot_confint(self, X: np.array, Y: np.array,
                        n_boot_samples: int = 5000,
                        metric: Optional[Callable[[Any], float]] = None) -> int:
        """
        Perform bootstrap confidence interval
        :param X: Null hypothesis distribution
        :param Y: Alternative hypothesis distribution
        :param metric: Custom metric (mean, median, percentile (1, 2, ...), etc
        :returns: Ratio of rejected H0 hypotheses to number of all tests
        """
        if X is None or Y is None:
            X = self.dataset.loc[self.dataset[self.__group_col] == 'A', self.__target].to_numpy()
            Y = self.dataset.loc[self.dataset[self.__group_col] == 'B', self.__target].to_numpy()

        metric_diffs: List[float] = []
        for _ in tqdm(range(n_boot_samples)):
            x_boot = np.random.choice(X, size=X.shape[0], replace=True)
            y_boot = np.random.choice(Y, size=Y.shape[0], replace=True)
            metric_diffs.append( metric(x_boot) - metric(y_boot) )
        pd_metric_diffs = pd.DataFrame(metric_diffs)

        left_quant = self.__alpha / 2
        right_quant = 1 - self.__alpha / 2
        ci = pd_metric_diffs.quantile([left_quant, right_quant])
        ci_left, ci_right = float(ci.iloc[0]), float(ci.iloc[1])

        test_result: int = 0 # 0 - cannot reject H0, 1 - reject H0
        if ci_left > 0 or ci_right < 0: # left border of ci > 0 or right border of ci < 0
            test_result = 1

        return test_result

    def test_boot_hypothesis(self, X: np.array, Y: np.array,
                             n_boot_samples: int = 5000,
                             use_correction: bool = False) -> float:
        """
        Perform T-test for independent samples with unequal number of observations and variance
        :param X: Null hypothesis distribution
        :param Y: Alternative hypothesis distribution
        :returns: Ratio of rejected H0 hypotheses to number of all tests
        """
        if X is None or Y is None:
            X = self.dataset.loc[self.dataset[self.__group_col] == 'A', self.__target].to_numpy()
            Y = self.dataset.loc[self.dataset[self.__group_col] == 'B', self.__target].to_numpy()

        T: int = 0
        for _ in range(n_boot_samples):
            x_boot = np.random.choice(X, size=X.shape[0], replace=True)
            y_boot = np.random.choice(Y, size=Y.shape[0], replace=True)

            T_boot = (np.mean(x_boot) - np.mean(y_boot)) / (np.var(x_boot) / x_boot.shape[0] + np.var(y_boot) / y_boot.shape[0])
            test_res = ttest_ind(x_boot, y_boot, equal_var=False, alternative=self.__alternative)

            if (use_correction and (T_boot >= (test_res[1] / n_boot_samples))) or \
                    (not use_correction and (T_boot >= test_res[1])):
                T += 1

        pvalue = T / n_boot_samples

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
        control_share, treatment_share = split_ratios if split_ratios is not None else self.__split_ratios
        if treatment_share == 0.5:
            alpha: float = (1 - self.__alpha / 2) if self.__alternative == 'two-sided' else (1 - self.__alpha)
            n_samples: int = round(2 * (t.ppf(alpha) + t.ppf(1 - self.__beta)) * std ** 2 / (effect_size ** 2), 0) + 1
            return (n_samples, n_samples)
        else:
            alpha: float = (1 - self.__alpha / 2) if self.__alternative == 'two-sided' else (1 - self.__alpha)
            n: int = round((((t.ppf(alpha) + t.ppf(1 - self.__beta)) * std ** 2 / (effect_size ** 2))) \
                      / (treatment_share * control_share), 0) + 1
            a_samples, b_samples = round(n * control_share, 0) + 1, round(n * treatment_share, 0) + 1
        return (a_samples, b_samples)

    def mde(self, std: float = None, n_samples: int = None) -> float:
        """
        Calculate Minimum Detectable Effect using Margin of Error formula
        :param std: Pooled standard deviatioin
        :param n_samples: Number of samples for each group
        :return: MDE, in absolute lift
        """
        alpha: float = (1 - self.__alpha / 2) if self.__alternative == 'two-sided' else (1 - self.__alpha)
        mde: float = np.sqrt( 2 * (t.ppf(alpha) + t.ppf(1 - self.__beta)) * std / n_samples )
        return mde

    def mde_hyperopt(self, n_iter: int = 20000, strategy: str = 'simple_test', params: Dict[str, List[float]] = None,
                     to_csv: bool = False, csv_path: str = None) -> None:
        def objective(params) -> float:
            split_rate, inc = params['split_rate'], params['inc']
            self._split_data(split_rate)
            control, treatment = self.dataset.loc[self.dataset[self.__group_col] == 'A', self.__target].to_numpy(), \
                                 self.dataset.loc[self.dataset[self.__group_col] == 'B', self.__target].to_numpy()
            treatment = self._add_increment('solid', treatment, inc)
            pvalue_mean = 0
            for it in range(n_iter):
                pvalue_mean += self.test_hypothesis(control, treatment)
            pvalue_mean /= n_iter
            return -pvalue_mean

        space = {}
        for param, values in params.items():
            space[param] = hp.uniform(param, values[0], values[1])
            # space[param] = hp.choice(param, )

        trials = Trials()
        print('\nSpace')
        print(space)
        best = fmin(objective,
                    space,
                    algo=tpe.suggest,
                    max_evals=10,
                    trials=trials
                    )
        print('\nBest')
        print(best)

        # Get the values of the optimal parameters
        best_params = space_eval(space, best)
        print('\nBest params')
        print(best_params)

    def plot(self) -> None:
        gr = Graphics()
        a = self.dataset.loc[self.dataset[self.__group_col] == 'A', self.__target].to_numpy()
        b = self.dataset.loc[self.dataset[self.__group_col] == 'B', self.__target].to_numpy()
        gr.plot_experiment(a, b, self.__alternative, self.__metric_name, self.__alpha, self.__beta)

    def cuped(self):
        vr = VarianceReduction()
        self.dataset = vr.cuped(self.dataset,
                                target=self.__target,
                                groups=self.__group_col,
                                covariate=self.__covariate)
        return self

    def cupac(self):
        vr = VarianceReduction()
        self.dataset = vr.cupac(self.dataset,
                               target_prev=self.__target_prev,
                               target_now=self.__target,
                               factors_prev=self.__predictors_prev,
                               factors_now=self.__predictors,
                               groups=self.__group_col)
        return self

    def __metric_calc(self, X: Union[List[Any], np.array]):
        if self.metric_name == 'mean':
            return np.mean(X)
        elif self.metric_name == 'median':
            return np.median(X)
        elif self.metric_name == 'custom':
            return self.metric(X)

    def __bucketize(self, X: pd.DataFrame):
        np.random.shuffle(X)
        X_new = np.array([ self.__metric_calc(x) for x in np.array_split(X, self.n_buckets) ])
        return X_new

    def bucketing(self):
        self.__control = self.__bucketize(self.__control)
        self.__treatment = self.__bucketize(self.__treatment)
        return self


if __name__ == '__main__':
    data = pd.DataFrame({
        'id': range(1, 10_002),
        'group': np.random.choice(a=['A', 'B'], size=10_001),
        'cheque': np.random.beta(a=2, b=8, size=10_001)
    })

    ab = ABTest()
    ab.use_dataset(data)
    ab.plot()