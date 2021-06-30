import numpy as np
import pandas as pd
import statsmodels.stats.api as sms
import math, os
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
from scipy.stats import mannwhitneyu, ttest_ind, kstwo
from typing import Dict, List, Tuple, Any, Union, Optional, Callable


class Splitter:
    def __init__(self) -> None:
        pass

    def set_splitter(self, splitter: Callable[[], Tuple[np.array, np.array]]):
        pass

    def _split(self, X: pd.DataFrame, confounding: List[str] = None):
        """
        Splitting of the dataset
        :param X: Dataframe for splitting
        :param confounding: List of confounding variables
        :return: Initial dataframe with one more column: group
        """
        X = X.sample(frac=1).reset_index(drop=True)
        X['group'] = ''
        X_unique_conf = X.groupby(by=confounding).group.count().reset_index()
        X_unique_conf.rename(columns={'group': 'cnt_obs'}, inplace=True)
        for _, row in X_unique_conf.iterrows():
            A_len = int(round(row['cnt_obs'] / 2))
            B_len = row['cnt_obs'] - A_len
            group_flag = ['A' for _ in range(A_len)] + ['B' for _ in range(B_len)]
            logicals = []
            for col in row.index.drop(labels=['cnt_obs']):
                logicals.append(np.array(X[col] == row[col]))
            X.loc[np.logical_and.reduce(logicals), 'group'] = group_flag
        return X

    def _stratify(self, X: pd.DataFrame, by: List[str] = None) -> pd.DataFrame:
        """
        Stratification of the dataset
        :param X: Dataframe for stratification
        :param by: List of columns on which to be stratified
        :return: Initial dataframe with one more column: strata
        """
        X['strata'] = 0
        X_unique_strata = X[by].drop_duplicates().reset_index(drop=True)
        for i, row in X_unique_strata.iterrows():
            logicals = []
            for col in row.index:
                logicals.append(np.array(X[col] == row[col]))
            X.loc[np.logical_and.reduce(logicals), 'strata'] = i
        return X

    def fit(self, X: pd.DataFrame, split_rate: float = 0.5, confounding: List[str] = None, stratify: bool = False, stratify_by: List[str] = None) -> pd.DataFrame:
        if stratify:
            X = self._stratify(X, by=stratify_by)
            Xs = []
            for strata in X.strata.unique():
                X_strata = X.loc[X.strata == strata]
                res = self._split(X_strata, confounding)
                Xs.append(res)
            return Xs
        else:
            X = self._split(X, confounding)
            return X


if __name__ == '__main__':
    # X = pd.read_csv('../data/external/bfp_15w.csv', sep=';', decimal=',')
    X = pd.DataFrame({
        'sex': ['f' for _ in range(14)] + ['m' for _ in range(6)],
        'married': ['yes' for _ in range(5)] + ['no' for _ in range(9)] + ['yes' for _ in range(4)] + ['no', 'no'],
        'country': [np.random.choice(['UK', 'US'], 1)[0] for _ in range(20)],
    })
    conf = ['sex', 'married']
    X_out = Splitter().fit(X, confounding=conf, stratify=True, stratify_by=['country'])
    for df in X_out:
        print(df.sort_values(by=conf))
