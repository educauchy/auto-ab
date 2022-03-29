import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Union, Callable, Optional, Any
import hashlib, pprint
from collections import Counter, defaultdict
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans, DBSCAN
from scipy.stats import ks_2samp, entropy
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score


class Splitter:
    def __init__(self, split_rate: float = 0.5,
                custom_splitter: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None) -> None:
        """
        :param split_rate: Share of control group
        :param custom_splitter: Custom splitter function which must take parameters:
            X: Pandas DataFrame
            target: Target column name (if continuous metric)
            numerator: Numerator column name (if ratio metric)
            denominator: Denominator column name (if ratio metric)
            split_rate: Split rate
        """
        self.split_rate = split_rate
        self.custom_splitter = custom_splitter

    def __clustering(self, X_full: pd.DataFrame = None, X: pd.DataFrame = None, n_clusters: int = None) -> np.array:
        """
        Clustering for dataset
        :param X_full: Pandas DataFrame to cluster
        :param X: Pandas DataFrame to predict cluster_id
        :param n_clusters: Number of clusters
        :return: Numpy array with cluster id
        """
        kmeans = KMeans(n_clusters=n_clusters)
        kmeans.fit(X_full)
        cluster_id = kmeans.predict(X)

        return cluster_id

    def __kl_divergence(self, a_cluster_id: Union[np.array, List[int]] = None,
                        b_cluster_id: Union[np.array, List[int]] = None, n_bins: int = 50) -> Tuple[float, float]:
        """
        Kullback-Leibler divergence for two arrays of cluster ids for A/A test
        :param a_cluster_id: List of cluster ids for group A
        :param b_cluster_id: List of cluster ids for group B
        :param n_bins: Number of clusters
        :return: Kullback-Leibler divergence a to b and b to a
        """
        a = a_cluster_id
        b = b_cluster_id

        bins_arr = list(range(1, n_bins+1))
        a = np.histogram(a, bins=bins_arr, density=True)[0]
        b = np.histogram(b, bins=bins_arr, density=True)[0]

        ent_ab = entropy(a, b)
        ent_ba = entropy(b, a)

        return (ent_ab, ent_ba)

    def __model_classify(self, X: pd.DataFrame = None, target: np.array = None) -> float:
        """
        Classification model for A/A test
        :param X: Dataset to classify
        :param target: Target for model
        :return: ROC-AUC score for model
        """
        clf = RandomForestClassifier()
        clf.fit(X, target)
        pred = clf.predict_proba(X)
        roc_auc = roc_auc_score(target, pred)

        return roc_auc

    def __alpha_simulation(self, X: pd.DataFrame = None, target: np.array = None,
                numerator: str = None, denominator: str = None,
                metric_type: str = 'solid',
                alpha: float = 0.05, n_iter: int = 10000) -> float:
        """
        Perform A/A test
        :param X: Pandas DataFrame to test
        :param target: Target column name (if continuous metric)
        :numerator: Numerator column name (if ratio metric)
        :denominator: Denominator column name (if ratio metric)
        :param metric_type: Test metric type
        :param alpha: Significance level
        :param n_iter: Number of iterations
        :return: Actual alpha; Share of iterations when control and treatment groups are equal
        """
        result: int = 0
        for it in range(n_iter):
            if metric_type == 'solid':
                X = self.fit(X, target=target, split_rate=self.split_rate)
                control, treatment = X.loc[X['group'] == 'A', target].to_numpy(), \
                                     X.loc[X['group'] == 'B', target].to_numpy()
                _, pvalue = ks_2samp(control, treatment)
                if pvalue >= alpha:
                    result += 1
            elif metric_type == 'ratio':
                X = self.fit(X, numerator=numerator, denominator=denominator, split_rate=self.split_rate)
                num_control, num_treatment = X.loc[X['group'] == 'A', numerator].to_numpy(), \
                                                X.loc[X['group'] == 'B', numerator].to_numpy()
                _, num_pvalue = ks_2samp(num_control, num_treatment)

                den_control, den_treatment = X.loc[X['group'] == 'A', denominator].to_numpy(), \
                                             X.loc[X['group'] == 'B', denominator].to_numpy()
                _, den_pvalue = ks_2samp(den_control, den_treatment)

                if num_pvalue >= alpha and den_pvalue >= alpha:
                    result += 1

        result /= n_iter

        return result

    def aa_test(self, X_full: pd.DataFrame = None, X: pd.DataFrame = None, target: np.array = None,
                group_id: np.array = None, method: str = None):
        if method == 'alpha-simulation':
            actual_alpha = self.__alpha_simulation(X, target)
            return actual_alpha
        elif method == 'kl-divergence':
            X_ext = X.copy()
            X_ext['group_id'] = group_id
            X_ext['cluster_id'] = self.__clustering(X_full, X, 50)
            a = X_ext.loc[X_ext.group_id == 'CONTROL', 'cluster_id'].tolist()
            b = X_ext.loc[X_ext.group_id == 'TREATMENT', 'cluster_id'].tolist()
            kl_divs = self.__kl_divergence(a, b, 50)
            return kl_divs
        elif method == 'model-classify':
            roc_auc = self.__model_classify(X, target)
            return roc_auc

    def fit(self, X: pd.DataFrame, target: np.array = None, numerator: str = None,
            denominator: str = None, split_rate: float = None) -> pd.DataFrame:
        """
        Split DataFrame and add group column based on splitting
        :param X: Pandas DataFrame to split
        :param split_rate: Split rate of control to treatment
        :return: DataFrame with additional 'group' column
        """
        if self.custom_splitter is None:
            split_rate = split_rate if split_rate is not None else self.split_rate
            A_data, B_data = train_test_split(X, train_size=split_rate, random_state=0)
            A_data.loc[:, 'group'] = 'A'
            B_data.loc[:, 'group'] = 'B'
            Z = pd.concat([A_data, B_data]).reset_index(drop=True)
        else:
            Z = self.custom_splitter(X, target, split_rate)

        return Z

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
    X = pd.DataFrame({
        'sex': ['f' for _ in range(14)] + ['m' for _ in range(6)],
        'married': ['yes' for _ in range(5)] + ['no' for _ in range(9)] + ['yes' for _ in range(4)] + ['no', 'no'],
        'country': [np.random.choice(['UK', 'US'], 1)[0] for _ in range(20)],
    })
    conf = ['sex', 'married']
    stratify_by = ['country']
    X_out = Splitter(split_rate=0.4, confounding=conf).fit(X)


