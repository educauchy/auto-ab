import numpy as np
import pandas as pd
import statsmodels.stats.api as sms
import math, os
from collections import Counter, defaultdict
from scipy.stats import mannwhitneyu, ttest_ind, shapiro, mode
from typing import Dict, List, Any, Union, Optional, Callable
from tqdm.auto import tqdm
from .splitter import Splitter


class ABTest:
    """Perform AB-test"""
    def __init__(self, alpha: float = 0.05, alternative: str = 'one-sided') -> None:
        self.alpha = alpha                  # use self.__alpha everywhere in the class
        self.alternative = alternative      # use self.__alternative everywhere in the class
        self.dataset: pd.DataFrame = None

    @property
    def alpha(self):
        return self.__alpha

    @alpha.setter
    def alpha(self, value):
        if 0 <= value <= 1:
            self.__alpha = value
        else:
            raise Exception('Significance level must be inside interval [0, 1]. Your input: {}.'.format(value))

    @property
    def alternative(self):
        return self.__alternative

    @alternative.setter
    def alternative(self, value):
        if value in ['less', 'greater', 'two-sided']:
            self.__alternative = value
        else:
            raise Exception("Alternative must be either 'less', 'greater', or 'two-sided'. Your input: '{}'.".format(value))

    def __str__(self):
        return f"ABTest(alpha={self.__alpha}, alternative='{self.__alternative}')"

    def _add_increment(self, X: np.array, inc_value: Union[float, int]) -> np.array:
        """
        Add constant increment to a list
        :param X: Numpy array to modify
        :param inc_value: Constant addendum to each value
        :returns: Modified X array
        """
        return X + inc_value

    def _split_data(self, X: pd.DataFrame, split_rate: float) -> pd.DataFrame:
        """
        Split data and add group column
        :param X: Pandas DataFrame to split
        :param split_rate: Split rate of control/treatment
        :return: DataFrame with additional 'group' column
        """
        X_with_groups = self.splitter.fit(X, split_rate)
        return X_with_groups

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

    def set_increment(self, inc_var: List[float] = None, extra_params: Dict[str, float] = None) -> None:
        self.increment_list = inc_var
        self.increment_extra = extra_params

    def set_split_rates(self, split_rates: List[float] = None) -> None:
        self.split_rates = split_rates

    def set_splitter(self, splitter: Splitter) -> None:
        """
        Add splitter
        :param splitter: Splitter instance of class Splitter
        """
        self.splitter = splitter

    def use_dataset(self, X: np.array, target: str = None) -> None:
        """
        Put dataset for analysis
        :param X: Dataset for analysis
        :param target: Target column name
        """
        self.dataset = X
        self.target = target

    def load_dataset(self, path: str = '', target: str = None) -> None:
        """
        Load dataset for analysis
        :param path: Path to the dataset for analysis
        :param target: Target column name
        """
        self.dataset = self._read_file(path)
        self.target = target

    def delta_method(self, Z: pd.DataFrame, numerator: str = '', denominator: str = '') -> int:
        pass

    def linearization(self, Z: pd.DataFrame, numerator: str = '', denominator: str = '') -> pd.DataFrame:
        """
        Important: there is an assumption that all data is already grouped by user
        :param Z: Pandas DataFrame for analysis
        :param numerator: Column name of numerator of ratio metric
        :param denominator: Column name of denominator of ratio metric
        :return: DataFrame with additional column 'lnrzd_metric'
        """
        X = Z.loc[Z['group'] == 'A']
        K = round(sum(X[numerator]) / sum(X[denominator]), 4)
        Z.loc[:, 'lnrzd_metric'] = Z[numerator] - K * Z[denominator]
        return Z

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

    def test_hypothesis_strat_confint(self, Z: pd.DataFrame,
                            metric: Optional[Callable[[Any], float]] = None,
                            strata_col: str = '',
                            weights: Dict[str, float] = None) -> int:
        """
        Perform stratification with confidence interval
        :param Z: Pandas DataFrame for analysis
        :param metric: Custom metric (mean, median, percentile (1, 2, ...), etc
        :param strata_col: Column name of strata column
        :return: Test result: 1 - significant different, 0 - insignificant difference
        """
        metric_diffs: List[float] = []
        X = Z.loc[Z['group'] == 'A']
        Y = Z.loc[Z['group'] == 'B']
        for _ in tqdm(range(self.n_boot_samples)):
            x_strata_metric = 0
            y_strata_metric = 0
            for strat in weights.keys():
                X_strata = X.loc[X[strata_col] == strat, self.target]
                Y_strata = Y.loc[Y[strata_col] == strat, self.target]
                x_strata_metric += (metric(np.random.choice(X_strata, size=X_strata.size // 2, replace=False)) * weights[strat])
                y_strata_metric += (metric(np.random.choice(Y_strata, size=Y_strata.size // 2, replace=False)) * weights[strat])
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
            x_boot = np.random.choice(X, size=X.size, replace=True)
            y_boot = np.random.choice(Y, size=Y.size, replace=True)
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
            x_boot = np.random.choice(X, size=X.size, replace=True)
            y_boot = np.random.choice(Y, size=Y.size, replace=True)
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
            x_boot = np.random.choice(X, size=X.size, replace=True)
            y_boot = np.random.choice(Y, size=Y.size, replace=True)

            T_boot = (np.mean(x_boot) - np.mean(y_boot)) / (np.var(x_boot) / x_boot.size + np.var(y_boot) / y_boot.size)
            test_res = ttest_ind(x_boot, y_boot, equal_var=False, alternative=self.__alternative)

            if (use_correction and (T_boot >= (test_res[1] / self.n_boot_samples))) or \
                    (not use_correction and (T_boot >= test_res[1])):
                T += 1

        pvalue = T / self.n_boot_samples
        return pvalue

    def mde_simulation(self, n_iter: int = 20000, strategy: str = 'means', strata: Optional[str] = '',
            n_boot_samples: Optional[int] = 10000, n_buckets: Optional[int] = None,
            metric: Optional[Callable[[Any], float]] = None, weights: Optional[Dict[str, float]] = None,
            use_correction: bool = True, to_csv: bool = False, csv_path: str = None) -> Dict[float, Dict[float, float]]:
        """
        Simulation process of determining appropriate split rate and increment rate for experiment
        :param n_iter: Number of iterations of simulation
        :param strategy: Name of strategy to use in experiment assessment
        :param strata: Strata column
        :param n_boot_samples: Number of bootstrap samples
        :param n_buckets: Number of buckets
        :param metric: Custom metric (mean, median, percentile (1, 2, ...), etc
        :param weights: Pre-experiment weights for strata column
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
                for it in range(n_iter):
                    print(f'Split_rate: {split_rate}, inc_rate: {inc}, iter: {it}')
                    X_with_groups = self._split_data(self.dataset, split_rate)
                    control, treatment = X_with_groups.loc[X_with_groups['group'] == 'A', self.target].to_numpy(), \
                                         X_with_groups.loc[X_with_groups['group'] == 'B', self.target].to_numpy()
                    treatment = self._add_increment(treatment, inc)

                    if strategy == 'simple_test':
                        test_result: int = self.test_hypothesis(control, treatment)
                        imitation_log[split_rate][inc] += test_result
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
                        test_result: int = self.test_hypothesis_strat_confint(X_with_groups, metric=metric, strata_col=strata, weights=weights)
                        imitation_log[split_rate][inc] += test_result
                    elif strategy == 'buckets':
                        test_result: int = self.test_hypothesis_buckets(control, treatment, metric=metric, n_buckets=n_buckets)
                        imitation_log[split_rate][inc] += test_result

                imitation_log[split_rate][inc] /= n_iter

                row = pd.DataFrame({
                    'split_rate': [split_rate],
                    'increment': [inc],
                    'pval_sign_share': [imitation_log[split_rate][inc]]})
                csv_pd = csv_pd.append(row)

        if to_csv:
            csv_pd.to_csv(csv_path, index=False)
        return dict(imitation_log)
