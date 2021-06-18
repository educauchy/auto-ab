import numpy as np
from auto_ab import ABTest
from typing import Dict, List, Tuple, Any, Union


class TestSplitter:
    def test_goodness_fit(self):
        def splitter_function(X: np.array) -> Tuple[Any, Any]:
            pass

        m = ABTest(alpha=0.05, alternative='two-sided')
        X = np.random.randint(0, 100, 100)
        m.use_dataset(X)
        m.set_increment(inc_var=[0])
        m.set_split_rate(split_rates=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
        m.set_splitter(splitter_function)
        m.mde(n_iter=10, n_boot_samples=10, to_csv=True, csv_name='pytest_splitter')

