from auto_ab import ABTest
import numpy as np
from typing import Dict, List, Tuple, Any, Union


class TestSplitter:
    def test_goodness_fit(self):
        def splitter_function(X: np.array) -> Tuple[Any, Any]:
            pass
        def metric(X: np.array) -> float:
            # return np.quantile(X, 0.1) # check while test fails on this metric
            return np.mean(X)

        m = ABTest(alpha=0.05, alternative='less')
        X = np.random.exponential(2, 100000)
        m.use_dataset(X)
        m.set_increment(inc_var=[0])
        m.set_split_rate(split_rates=[0.1, 0.2, 0.3, 0.4, 0.5])
        m.set_splitter(splitter_function)
        m.mde(n_iter=10, n_boot_samples=50, metric=metric, n_buckets=100, test_type='buckets', to_csv=False,
              csv_name='./auto_ab/data/splitter_10quantile_buckets.csv')
