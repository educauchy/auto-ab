import numpy as np
import pandas as pd
import math, os
from collections import Counter, defaultdict
from scipy.stats import mannwhitneyu, ttest_ind, shapiro, mode, t
from typing import Dict, List, Any, Union, Optional, Callable, Tuple
from tqdm.auto import tqdm
from .splitter import Splitter
from hyperopt import hp, fmin, tpe, Trials, space_eval
from .pulse import Pulse


class ABTest:
    """Perform AB-test"""
    def __init__(self, alpha: float = 0.05, beta: float = 0.80,
                 alternative: str = 'two-sided') -> None:
        self.alpha = alpha                  # use self.__alpha everywhere in the class
        self.beta = beta                    # use self.__beta everywhere in the class
        self.alternative = alternative      # use self.__alternative everywhere in the class
        self.target: Optional[str] = None
        self.dataset: pd.DataFrame = None
        self.initial_dataset: pd.DataFrame = None    # for ratio metrics to keep old dataset
        self.splitter: Splitter = None
        self.split_rates: List[float] = None
        self.increment_list: List[float] = None
        self.increment_extra: Dict[str, float] = None

    @property
    def alpha(self):
        return self.__alpha

    @alpha.setter
    def alpha(self, value: float):
        if 0 <= value <= 1:
            self.__alpha = value
        else:
            raise Exception('Significance level must be inside interval [0, 1]. Your input: {}.'.format(value))

    @property
    def beta(self):
        return self.__beta

    @beta.setter
    def beta(self, value: float):
        if 0 <= value <= 1:
            self.__beta = value
        else:
            raise Exception('Power must be inside interval [0, 1]. Your input: {}.'.format(value))

    # @property
    # def split_rates(self) -> List[Union[float, int]]:
    #     return self.split_ratex
    #
    # @split_rates.setter
    # def split_rates(self, value: List[Union[float, int]]) -> None:
    #     if isinstance(value, list) and len(value) > 0:
    #         self.split_rates = sorted(value, reverse=True)
    #     else:
    #         raise Exception('Split rates must be a list. Your input: {}.'.format(value))

    @property
    def alternative(self) -> str:
        return self.__alternative

    @alternative.setter
    def alternative(self, value: float) -> None:
        if value in ['less', 'greater', 'two-sided']:
            self.__alternative = value
        else:
            raise Exception("Alternative must be either 'less', 'greater', or 'two-sided'. Your input: '{}'.".format(value))

    def __str__(self):
        return f"ABTest(alpha={self.__alpha}, alternative='{self.__alternative}')"

    def _add_increment(self, metric_type: str = None, X: Union[pd.DataFrame, np.array] = None,
                       inc_value: Union[float, int] = None) -> np.array:
        """
        Add constant increment to a list
        :param X: Numpy array to modify
        :param inc_value: Constant addendum to each value
        :returns: Modified X array
        """
        if metric_type == 'solid':
            return X + inc_value
        elif metric_type == 'ratio':
            X.loc[:, 'inced'] = X[self.numerator] + inc_value
            X.loc[:, 'diff'] = X[self.denominator] - X[self.numerator]
            X.loc[:, 'rand_inc'] = np.random.randint(0, X['diff'] + 1, X.shape[0])
            X.loc[:, 'numerator_new'] = X[self.numerator] + X['rand_inc']

            X[self.numerator] = np.where(X['inced'] < X[self.denominator], X['inced'], X['numerator_new'])
            return X[[self.numerator, self.denominator]]

    def _split_data(self, split_rate: float) -> None:
        """
        Add 'group' column
        :param split_rate: Split rate of control/treatment
        :return: None
        """
        self.dataset = self.splitter.fit(self.dataset, self.target, self.numerator, self.denominator, split_rate)

    def _read_file(self, path: str) -> pd.DataFrame:
        """
        Read file and return pandas dataframe
        :param path: Path to file
        :returns: Pandas DataFrame
        """
        df = None
        _, file_ext = os.path.splitext(path)
        if file_ext == '.csv':
            df = pd.read_csv(path, encoding='utf8')
        elif file_ext == '.xls' or file_ext == '.xlsx':
            df = pd.read_excel(path, encoding='utf8')
        return df

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

    def _linearize(self, numerator: str = '', denominator: str = ''):
            X = self.dataset.loc[self.dataset['group'] == 'A']
            K = round(sum(X[numerator]) / sum(X[denominator]), 4)
            self.dataset.loc[:, f'{numerator}_{denominator}'] = self.dataset[numerator] - K * self.dataset[denominator]
            self.target = f'{numerator}_{denominator}'

    def _delta_params(self, df: pd.DataFrame, numerator: str, denominator: str = '') -> Tuple[float, float]:
        """
        Calculated expectation and variance for ratio metric using delta approximation
        :param df: Pandas DataFrame of particular group (A, B, etc)
        :param numerator: Ratio numerator column name
        :param denominator: Ratio denominator column name
        :return: Tuple with mean and variance of ratio
        """
        num = df[numerator]
        den = df[denominator]
        num_mean = num.mean()
        num_var = num.var()
        den_mean = den.mean()
        den_var = den.var()
        cov = df[[numerator, denominator]].cov()[0, 1]
        n = len(num)

        bias_correction = (den_mean / num_mean ** 3) * (num_var / n) - cov / (n * num_mean ** 2)
        mean = den_mean / num_mean - 1 + bias_correction
        var = den_var / num_mean ** 2 - 2 * (den_mean / num_mean ** 3) * cov + (den_mean ** 2 / num_mean ** 4) * num_var

        return (mean, var)

    def _taylor_params(self, df: pd.DataFrame, numerator: str, denominator: str = '') -> Tuple[float, float]:
        """
        Calculated expectation and variance for ratio metric using Taylor expansion approximation
        :param df: Pandas DataFrame of particular group (A, B, etc)
        :param numerator: Ratio numerator column name
        :param denominator: Ratio denominator column name
        :return: Tuple with mean and variance of ratio
        """
        num = df[numerator]
        den = df[denominator]
        mean = num.mean() / den.mean() - df[[numerator, denominator]].cov()[0, 1] / (den.mean() ** 2) + den.var() * num.mean() / (den.mean() ** 3)
        var = (num.mean() ** 2) / (den.mean() ** 2) * (num.var() / (num.mean() ** 2) - 2 * df[[numerator, denominator]].cov()[0, 1]) / (num.mean() * den.mean() + den.var() / (den.mean() ** 2))

        return (mean, var)

    def set_increment(self, inc_var: List[float] = None, extra_params: Dict[str, float] = None) -> None:
        self.increment_list = inc_var
        self.increment_extra = extra_params

    def use_dataset(self, X: pd.DataFrame, id_col: str = None, target: Optional[str] = None,
                    numerator: Optional[str] = None, denominator: Optional[str] = None) -> None:
        """
        Put dataset for analysis
        :param X: Pandas DataFrame for analysis
        :param id_col: Id column name
        :param target: Target column name
        :param numerator: Ratio numerator column name
        :param denominator: Ratio denominator column name
        """
        self.dataset = X
        self.id = id_col
        self.target = target
        self.numerator = numerator
        self.denominator = denominator

    def load_dataset(self, path: str = '', id_col: str = None, target: Optional[str] = None,
                     numerator: Optional[str] = None, denominator: Optional[str] = None) -> None:
        """
        Load dataset for analysis
        :param path: Path to the dataset for analysis
        :param id_col: Id column name
        :param target: Target column name
        :param numerator: Ratio numerator column name
        :param denominator: Ratio denominator column name
        """
        self.dataset = self._read_file(path)
        self.id = id_col
        self.target = target
        self.numerator = numerator
        self.denominator = denominator

    def ratio_bootstrap(self) -> int:
        A = self.dataset[self.dataset.group == 'A']
        B = self.dataset[self.dataset.group == 'B']
        a_metric_total = sum(A[self.numerator]) / sum(A[self.denominator])
        b_metric_total = sum(B[self.numerator]) / sum(B[self.denominator])
        origin_mean = b_metric_total - a_metric_total
        boot_diffs = []
        boot_a_metric = []
        boot_b_metric = []

        for _ in tqdm(range(self.n_boot_samples)):
            a_boot = A[A[self.id].isin(A[self.id].sample(A[self.id].nunique(), replace=True))]
            b_boot = B[B[self.id].isin(B[self.id].sample(B[self.id].nunique(), replace=True))]
            a_boot_metric = sum(a_boot[self.numerator]) / sum(a_boot[self.denominator])
            b_boot_metric = sum(b_boot[self.numerator]) / sum(b_boot[self.denominator])
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
        A = self.dataset[self.dataset.group == 'A']
        B = self.dataset[self.dataset.group == 'B']

        A_mean, A_var = self._taylor_params(A, self.numerator, self.denominator)
        B_mean, B_var = self._taylor_params(B, self.numerator, self.denominator)
        test_result: int = self._manual_ttest(A_mean, A_var, A.shape[0], B_mean, B_var, B.shape[0])

        return test_result

    def delta_method(self) -> int:
        """
        Delta method with bias correction for ratios
        Source: https://arxiv.org/pdf/1803.06336.pdf
        :param numerator: Ratio numerator column name
        :param denominator: Ratio denominator column name
        :return: Hypothesis test result: 0 - cannot reject H0, 1 - reject H0
        """
        A = self.dataset[self.dataset.group == 'A']
        B = self.dataset[self.dataset.group == 'B']

        A_mean, A_var = self._delta_params(A, self.numerator, self.denominator)
        B_mean, B_var = self._delta_params(B, self.numerator, self.denominator)
        test_result: int = self._manual_ttest(A_mean, A_var, A.shape[0], B_mean, B_var, B.shape[0])

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
            not_ratio_columns = self.dataset.columns[~self.dataset.columns.isin([self.numerator, self.denominator])].tolist()
            df_grouped = self.dataset.groupby(by=not_ratio_columns, as_index=False).agg({
                self.numerator: 'sum',
                self.denominator: 'sum'
            })
            self.initial_dataset = self.dataset.copy(deep=True)
            self.dataset = df_grouped
        self._linearize(self.numerator, self.denominator)

    def test_hypothesis(self, X: np.array, Y: np.array) -> int:
        """
        Perform Welch's t-test / Mann-Whitney test for means/medians
        :param X: Group A
        :param Y: Group B
        :return: Test result: 0 - cannot reject H0, 1 - reject H0
        """
        test_result: int = 0
        if shapiro(X)[1] >= self.__alpha and shapiro(X)[1] >= self.__alpha:
            _, pvalue = ttest_ind(X, Y, equal_var=False, alternative=self.__alternative)
        else:
            _, pvalue = mannwhitneyu(X, Y, alternative=self.__alternative)
        if pvalue <= self.__alpha:
            test_result = 1

        return test_result

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
            test_result = self.test_hypothesis_boot_confint(X_new, Y_new, metric)

        return test_result

    def test_hypothesis_strat_confint(self, metric: Optional[Callable[[Any], float]] = None,
                            strata_col: str = '',
                            weights: Dict[str, float] = None) -> int:
        """
        Perform stratification with confidence interval
        :param metric: Custom metric (mean, median, percentile (1, 2, ...), etc
        :param strata_col: Column name of strata column
        :return: Test result: 1 - significant different, 0 - insignificant difference
        """
        metric_diffs: List[float] = []
        X = self.dataset.loc[self.dataset['group'] == 'A']
        Y = self.dataset.loc[self.dataset['group'] == 'B']
        for _ in tqdm(range(self.n_boot_samples)):
            x_strata_metric = 0
            y_strata_metric = 0
            for strat in weights.keys():
                X_strata = X.loc[X[strata_col] == strat, self.target]
                Y_strata = Y.loc[Y[strata_col] == strat, self.target]
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
                        metric: Optional[Callable[[Any], float]] = None) -> float:
        """
        Perform bootstrap confidence interval with
        :param X: Null hypothesis distribution
        :param Y: Alternative hypothesis distribution
        :param metric: Custom metric (mean, median, percentile (1, 2, ...), etc
        :returns: Type I error rate
        """
        metric_diffs: List[float] = []
        for _ in tqdm(range(self.n_boot_samples)):
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
                        metric: Optional[Callable[[Any], float]] = None) -> int:
        """
        Perform bootstrap confidence interval
        :param X: Null hypothesis distribution
        :param Y: Alternative hypothesis distribution
        :param metric: Custom metric (mean, median, percentile (1, 2, ...), etc
        :returns: Ratio of rejected H0 hypotheses to number of all tests
        """
        metric_diffs: List[float] = []
        for _ in tqdm(range(self.n_boot_samples)):
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

    def test_boot_hypothesis(self, X: np.array, Y: np.array, use_correction: bool = False) -> float:
        """
        Perform T-test for independent samples with unequal number of observations and variance
        :param X: Null hypothesis distribution
        :param Y: Alternative hypothesis distribution
        :returns: Ratio of rejected H0 hypotheses to number of all tests
        """
        T: int = 0
        for _ in range(self.n_boot_samples):
            x_boot = np.random.choice(X, size=X.shape[0], replace=True)
            y_boot = np.random.choice(Y, size=Y.shape[0], replace=True)

            T_boot = (np.mean(x_boot) - np.mean(y_boot)) / (np.var(x_boot) / x_boot.shape[0] + np.var(y_boot) / y_boot.shape[0])
            test_res = ttest_ind(x_boot, y_boot, equal_var=False, alternative=self.__alternative)

            if (use_correction and (T_boot >= (test_res[1] / self.n_boot_samples))) or \
                    (not use_correction and (T_boot >= test_res[1])):
                T += 1

        pvalue = T / self.n_boot_samples

        return pvalue

    def mde(self, control_mean: float, s_cont, s_treat, n_cont, n_treat) -> float:
        m = t.ppf(self.__alpha / 2) + t.ppf(self.__beta)
        s = np.sqrt()
        mde = m * s / control_mean
        pass

    def mde_simulation(self, n_iter: int = 20000, strategy: str = 'simple_test', strata: Optional[str] = '',
            metric_type: str = 'solid', n_boot_samples: Optional[int] = 10000, n_buckets: Optional[int] = None,
            metric: Optional[Callable[[Any], float]] = None, strata_weights: Optional[Dict[str, float]] = None,
            use_correction: Optional[bool] = True, to_csv: bool = False,
            csv_path: str = None) -> Dict[float, Dict[float, float]]:
        """
        Simulation process of determining appropriate split rate and increment rate for experiment
        :param n_iter: Number of iterations of simulation
        :param strategy: Name of strategy to use in experiment assessment
        :param strata: Strata column
        :param n_boot_samples: Number of bootstrap samples
        :param n_buckets: Number of buckets
        :param metric: Custom metric (mean, median, percentile (1, 2, ...), etc
        :param strata_weights: Pre-experiment weights for strata column
        :param use_correction: Whether or not to use correction for multiple tests
        :param to_csv: Whether or not to save result to a .csv file
        :param csv_path: CSV file path
        :return: Dict with ratio of hypotheses rejected under certain split rate and increment
        """
        if n_boot_samples < 1:
            raise Exception('Number of bootstrap samples must be 1 or more. Your input: {}.'.format(n_boot_samples))
        self.n_boot_samples = n_boot_samples
        imitation_log: Dict[float, Dict[float, int]] = defaultdict(float)
        csv_pd = pd.DataFrame()
        for split_rate in self.split_rates:
            imitation_log[split_rate] = {}
            for inc in self.increment_list:
                imitation_log[split_rate][inc] = 0
                curr_iter = 0
                for it in range(n_iter):
                    curr_iter += 1
                    print(f'Split_rate: {split_rate}, inc_rate: {inc}, iter: {it}')
                    self._split_data(split_rate)
                    if metric_type == 'solid':
                        control, treatment = self.dataset.loc[self.dataset['group'] == 'A', self.target].to_numpy(), \
                                             self.dataset.loc[self.dataset['group'] == 'B', self.target].to_numpy()
                        treatment = self._add_increment('solid', treatment, inc)
                    elif metric_type == 'ratio':
                        control, treatment = self.dataset.loc[self.dataset['group'] == 'A', [self.numerator, self.denominator]], \
                                             self.dataset.loc[self.dataset['group'] == 'B', [self.numerator, self.denominator]]
                        treatment = self._add_increment('ratio', treatment, inc)

                    if strategy == 'simple_test':
                        test_result: int = self.test_hypothesis(control, treatment)
                        imitation_log[split_rate][inc] += test_result
                    elif strategy == 'ratio_delta':
                        pass
                    elif strategy == 'ratio_taylor':
                        pass
                    elif strategy == 'boot_hypothesis':
                        pvalue: float = self.test_boot_hypothesis(control, treatment, use_correction=use_correction)
                        if pvalue <= self.__alpha:
                            imitation_log[split_rate][inc] += 1
                    elif strategy == 'boot_est':
                        pvalue: float = self.test_hypothesis_boot_est(control, treatment, metric=metric)
                        if pvalue <= self.__alpha:
                            imitation_log[split_rate][inc] += 1
                    elif strategy == 'boot_confint':
                        test_result: int = self.test_hypothesis_boot_confint(control, treatment, metric=metric)
                        imitation_log[split_rate][inc] += test_result
                    elif strategy == 'strata_confint':
                        test_result: int = self.test_hypothesis_strat_confint(metric=metric,
                                                                              strata_col=strata,
                                                                              weights=strata_weights)
                        imitation_log[split_rate][inc] += test_result
                    elif strategy == 'buckets':
                        test_result: int = self.test_hypothesis_buckets(control, treatment,
                                                                        metric=metric,
                                                                        n_buckets=n_buckets)
                        imitation_log[split_rate][inc] += test_result
                    elif strategy == 'pulse':
                        pulse = Pulse(control, treatment)
                        pvalue: float = pulse.is_similar()
                        if pvalue <= self.__alpha:
                            imitation_log[split_rate][inc] += 1

                    # do not need to proceed if already achieved desired level
                    if imitation_log[split_rate][inc] / curr_iter >= self.__beta:
                        break

                imitation_log[split_rate][inc] /= curr_iter

                row = pd.DataFrame({
                    'split_rate': [split_rate],
                    'increment': [inc],
                    'pval_sign_share': [imitation_log[split_rate][inc]]})
                csv_pd = csv_pd.append(row)

        if to_csv:
            csv_pd.to_csv(csv_path, index=False)
        return dict(imitation_log)

    def mde_hyperopt(self, n_iter: int = 20000, strategy: str = 'simple_test', params: Dict[str, List[float]] = None,
                     to_csv: bool = False, csv_path: str = None) -> None:
        def objective(params) -> float:
            split_rate, inc = params['split_rate'], params['inc']
            self._split_data(split_rate)
            control, treatment = self.dataset.loc[self.dataset['group'] == 'A', self.target].to_numpy(), \
                                 self.dataset.loc[self.dataset['group'] == 'B', self.target].to_numpy()
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
