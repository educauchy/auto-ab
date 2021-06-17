import numpy as np
import os, sys, yaml, json
from auto_ab import ABTest


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


# Example scenario to use
# x = np.random.normal(0, 1, 10000)
# y = np.random.normal(0, 1, 20000)
# m.use_datasets(x, y)

m = ABTest(alpha=0.05, alternative='two-sided')
X = np.random.randint(0, 300, 1000)
m.use_dataset(X)
m.set_increment(inc_var=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5])
m.set_split_rate(split_rates=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
m.set_splitter()
res = m.mde(n_iter=500, n_boot_samples=5, to_csv=True)
print(json.dumps(res, indent=4))

# m.generate_datasets(n_samples=config['n_samples'], dist1=config['dist1'], dist1_params=config['dist1_params'], \
#                     dist2=config['dist2'], dist2_params=config['dist2_params'])
# m.load_dataset('./data/test_dataset.csv', type='continuous', output='response', split_by='group', confound=None)
# m.plot_distributions(save_path='./output/AB_dists.png')
# print(m.power_analysis(n_samples=config['power']['n_samples'], effect_size=config['power']['effect_size'], power=None))
# m.run_simulation()

