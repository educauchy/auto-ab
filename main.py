import numpy as np
import pandas as pd
import os, sys, yaml, json
from auto_ab import ABTest, Splitter
from typing import Dict, List, Tuple, Any, Union


try:
    project_dir = os.path.dirname(__file__)
    config_file = os.path.join(project_dir, 'config.yaml')
    with open (config_file, 'r') as file:
        config = yaml.safe_load(file)
except yaml.YAMLError as exc:
    print(exc)
    sys.exit(1)
except Exception as e:
    print('Error reading the config file')
    sys.exit(1)




def metric(X: np.array) -> float:
    return np.quantile(X, 0.1)

# Data
mult = 5000
X = pd.DataFrame({
    'sex': (['f' for _ in range(14)] + ['m' for _ in range(6)]) * mult,
    'married': (['yes' for _ in range(5)] + ['no' for _ in range(9)] + ['yes' for _ in range(4)] + ['no', 'no']) * mult,
    'country': [np.random.choice(['UK', 'US'], 1)[0] for _ in range(20)] * mult,
    'height': [np.random.randint(160, 190, 1)[0] for _ in range(20)] * mult,
})
conf = ['sex', 'married']
stratify_by = ['country']
print(X)

# Splitter
splitter = Splitter(split_rate=0.5, confounding=conf, stratify=False, stratify_by=stratify_by)

# AB-test
m = ABTest(alpha=0.05, alternative='two-sided')
m.use_dataset(X, target='height')
m.set_increment(inc_var=[0.1, 0.2, 0.3, 0.4, 0.5, 0.8, 1, 2], extra_params={})
m.set_split_rate(split_rates=[0.1, 0.2, 0.3, 0.4, 0.5])
m.set_splitter(splitter)
res = m.mde(n_iter=10, n_buckets=200, metric=metric, test_type='buckets', to_csv=True, csv_name='./data/buckets_10quantile.csv')
print(json.dumps(res, indent=4))
