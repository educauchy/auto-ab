import numpy as np
import pandas as pd
from typing import List, Tuple, Union, Callable, Optional
import hashlib
from collections import Counter
import pprint
from sklearn.model_selection import train_test_split


class Splitter:
    def __init__(self, split_rate: float = 0.5, confounding: Optional[List[str]] = None) -> None:
        self.split_rate = split_rate
        self.confounding = confounding

    def set_splitter(self, splitter: Callable[[], Tuple[np.array, np.array]]):
        pass

    def _split(self, X: pd.DataFrame, target: str = '',split_rate: float = 0.5) -> pd.DataFrame:
        """
        Splitting of the dataset
        :param X: Dataframe for splitting
        :param split_rate: Split rate between control and treatment groups
        :param confounding: List of confounding variables
        :return: Initial dataframe with one more column: group
        """
        X_data = X[X.columns[~X.columns.isin([target])]]
        X_target = X[target]

        A_data, B_data, A_target, B_target = train_test_split(X_data, X_target, train_size=split_rate, random_state=0)
        A_data.loc[:, 'group'] = 'A'
        A_data.loc[:, target] = A_target
        B_data.loc[:, 'group'] = 'B'
        B_data.loc[:, target] = B_target
        Z = pd.concat([A_data, B_data]).reset_index(drop=True)

        return Z

    def fit(self, X: pd.DataFrame, target: str = '', split_rate: float = None) -> pd.DataFrame:
        """
        Splitting data
        :param X: Pandas DataFrame to split
        :param split_rate: Split rate of control to treatment
        :return: DataFrame with additional 'group' column
        """
        self.split_rate = split_rate if split_rate is not None else self.split_rate
        X = self._split(X, target, self.split_rate)
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


