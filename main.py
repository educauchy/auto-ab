import numpy as np
import pandas as pd
import os, sys, yaml, json
from auto_ab import ABTest, Splitter
import time
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
    # return np.median(X)

# Data
mult = config['mult']
X = pd.DataFrame({
    'id': list(range(0, 20 * mult)),
    'sex': (['f' for _ in range(14)] + ['m' for _ in range(6)]) * mult,
    'married': (['yes' for _ in range(5)] + ['no' for _ in range(9)] + ['yes' for _ in range(4)] + ['no', 'no']) * mult,
    'country': ['US' for _ in range(14 * mult)] + ['UK' for _ in range(6 * mult)],
    'height': np.concatenate([np.random.normal(190, 2, 14 * mult), np.random.normal(182, 3, 6 * mult)]),
})


# Splitter
splitter = Splitter(confounding=config['conf'], stratify=config['stratify'],\
                    stratify_by=config['stratify_by'], split_rate=config['split_rate'])

# AB-test
m = ABTest(alpha=config['alpha'], alternative=config['alternative'])
m.use_dataset(X, target=config['target'])
m.set_increment(inc_var=config['increment']['vars'], extra_params=config['increment']['extra_params'])
m.set_split_rate(split_rates=config['split_rates'])
m.set_splitter(splitter)


print('Strata begin')
start_time = time.time()
res = m.mde(n_iter=config['n_iter'], n_boot_samples=config['n_boot_samples'], metric=metric, strategy='strata_confint',
            strata=config['strata'], weights=config['weights'], to_csv=False, csv_name='./data/buckets_10quantile.csv')
print(json.dumps(res, indent=4))
print("--- %s seconds ---" % (time.time() - start_time))

print('Bootstrap begins')
start_time = time.time()
res = m.mde(n_iter=config['n_iter'], n_boot_samples=config['n_boot_samples'], metric=metric, strategy='boot_confint',
            to_csv=False, csv_name='./data/buckets_10quantile.csv')
print(json.dumps(res, indent=4))
print("--- %s seconds ---" % (time.time() - start_time))
