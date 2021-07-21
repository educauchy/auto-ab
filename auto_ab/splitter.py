import numpy as np
import pandas as pd
from typing import List, Tuple, Union, Callable
import hashlib
from collections import Counter
import pprint


class Splitter:
    def __init__(self, split_rate: float = 0.5, confounding: List[str] = None) -> None:
        self.split_rate = split_rate
        self.confounding = confounding

    def set_splitter(self, splitter: Callable[[], Tuple[np.array, np.array]]):
        pass

    def _split(self, X: pd.DataFrame, split_rate: float = 0.5, confounding: List[str] = None) -> pd.DataFrame:
        """
        Splitting of the dataset
        :param X: Dataframe for splitting
        :param split_rate: Split rate between control and treatment groups
        :param confounding: List of confounding variables
        :return: Initial dataframe with one more column: group
        """
        X = X.sample(frac=1).reset_index(drop=True)
        X['group'] = ''
        X_unique_conf = X.groupby(by=confounding).group.count().reset_index()
        X_unique_conf.rename(columns={'group': 'cnt_obs'}, inplace=True)
        for _, row in X_unique_conf.iterrows():
            A_len = int(round(row['cnt_obs'] * split_rate))
            B_len = row['cnt_obs'] - A_len
            group_flag = ['A' for _ in range(A_len)] + ['B' for _ in range(B_len)]
            logicals = []
            for col in row.index.drop(labels=['cnt_obs']):
                logicals.append(np.array(X[col] == row[col]))
            X.loc[np.logical_and.reduce(logicals), 'group'] = group_flag
        return X

    def fit(self, X: pd.DataFrame, split_rate: float = None) -> pd.DataFrame:
        self.split_rate = split_rate if split_rate is not None else self.split_rate
        X = self._split(X, self.split_rate, self.confounding)
        return X

    def create_level(self, X: pd.DataFrame, id_column: str = '', salt: Union[str, int] = '',
                     n_buckets: int = 100) -> pd.DataFrame:
        """
        Create new levels in split all users into buckets
        :param X: Pandas DataFrame
        :param id_column: User id column name
        :param salt: Salt string for the experiment
        :param n_buckets: Number of buckets for level
        :return: Pandas DataFrame extended by column 'bucket'
        """
        ids: np.array = X[id_column].to_numpy()
        salt: str = salt if type(salt) is str else str(int)
        salt: bytes = bytes(salt, 'utf-8')
        hasher = hashlib.blake2b(salt=salt)

        bucket_ids: np.array = np.array([])
        for id in ids:
            hasher.update( bytes(str(id), 'utf-8') )
            bucket_id = int(hasher.hexdigest(), 16) % n_buckets
            bucket_ids = np.append(bucket_ids, bucket_id)

        X.loc[:, 'bucket_id'] = bucket_ids
        X = X.astype({'bucket_id': 'int32'})
        return X


if __name__ == '__main__':
    # Test hash function
    X = pd.DataFrame({
        'id': range(0, 20000)
    })
    sp = Splitter()
    X_new = sp.create_level(X, 'id', 5, 200)
    level = X_new['bucket_id']
    pprint.pprint(Counter(level))

    # Test splitter
    # X = pd.read_csv('../data/external/bfp_15w.csv', sep=';', decimal=',')
    X = pd.DataFrame({
        'sex': ['f' for _ in range(14)] + ['m' for _ in range(6)],
        'married': ['yes' for _ in range(5)] + ['no' for _ in range(9)] + ['yes' for _ in range(4)] + ['no', 'no'],
        'country': [np.random.choice(['UK', 'US'], 1)[0] for _ in range(20)],
    })
    conf = ['sex', 'married']
    stratify_by = ['country']
    X_out = Splitter(split_rate=0.4, confounding=conf).fit(X)

