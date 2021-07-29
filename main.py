import numpy as np
import pandas as pd
import os, sys, yaml, json
from auto_ab import ABTest, Splitter


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
mult = config['mult']
sessions = np.random.randint(1, 10, 20 * mult)
X = pd.DataFrame({
    'id': np.random.choice(range(1, 100), 20 * mult),
    'sex': (['f' for _ in range(14)] + ['m' for _ in range(6)]) * mult,
    'married': (['yes' for _ in range(5)] + ['no' for _ in range(9)] + ['yes' for _ in range(4)] + ['no', 'no']) * mult,
    'country': ['US' for _ in range(14 * mult)] + ['UK' for _ in range(6 * mult)],
    'height': np.concatenate([np.random.normal(190, 2, 14 * mult), np.random.normal(182, 3, 6 * mult)]),
    'clicks': [np.random.choice(range(0, session+1), 1, replace=False)[0] for session in sessions],
    'sessions': sessions,
})
X.sort_values(by=['id'], inplace=True)

# X = pd.DataFrame({
#     'id':       [1, 1, 1, 1, 2, 2, 2, 3, 3, 4, 5, 5, 6, 6, 7, 8, 9],
#     'clicks':   [2, 4, 1, 9, 2, 6, 1, 7, 7, 9, 1, 1, 6, 5, 7, 8, 9],
#     'sessions': [10, 36, 5, 55, 9, 7, 3, 11, 14, 16, 3, 5, 11, 12, 8, 8, 9],
#     'group':    ['A', 'A', 'A', 'A', 'B', 'B', 'B', 'B', 'B', 'A', 'A', 'B', 'A', 'B', 'B', 'A', 'A'],
# })

splitter = Splitter(split_rate=config['splitter']['split_rate'])

m = ABTest(alpha=config['hypothesis']['alpha'], alternative=config['hypothesis']['alternative'])
m.use_dataset(X, id_col=config['data']['id_col'],
              numerator=config['data']['numerator'],
              denominator=config['data']['denominator'])
m.split_rates = config['simulation']['split_rates']
m.splitter = splitter
m.set_increment(inc_var=config['simulation']['increment']['vars'],
                extra_params=config['simulation']['increment']['extra_params'])
res = m.mde_simulation(n_iter=config['simulation']['n_iter'], n_boot_samples=config['hypothesis']['n_boot_samples'],
                       metric_type=config['metric']['metric_type'], metric=metric, strategy=config['hypothesis']['strategy'],
                       strata=config['hypothesis']['strata'], strata_weights=config['hypothesis']['strata_weights'],
                       to_csv=config['result']['to_csv'], csv_path=config['result']['csv_path'])
print(json.dumps(res, indent=4))
