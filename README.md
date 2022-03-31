# auto_ab - automation of A/B-experiments

**auto_ab** is a library developed to automate routine tasks during runs of multiple online controlled experiments, also known as A/B-tests.

## Current features

- Custom metrics
- Splitting
  - Splitter
  - Test splitter (A/A-test)
  - Custom split rate and increment rate
- Continuous metrics
  - Classical hypothesis testing
  - Bootstrap hypothesis testing with 3 different approaches
  - Bucketing hypothesis testing
- Ratio metrics
  - Delta method on ratios
  - Taylor expansion on ratios
  - Linearization on ratios
  - Bootstrap on ratios
- Variance reduction
  - CUPED
  - CUPAC


## Config file

Parameters for A/B-test are set in config file. The example of config file you can find below.

```yaml
metric:
  metric_type: 'ratio'
  metric_name: 'mean'
data:
  id_col:       'id_column_name'
  group_col:    'group_column_name'
  target:       'target_column_name'
  numerator:    'numerator_column_name'
  denominator:  'denominator_column_name'
splitter:
  run_aa: True
simulation:
  n_iter: 100
  split_rates: [0.1, 0.2, 0.3, 0.4, 0.5]
  increment:
    vars: [0, 1, 2, 3, 4, 5]
    extra_params: []
hypothesis:
  alpha: 0.05
  beta: 0.2
  alternative: 'two-sided'
  split_ratios: [0.5, 0.5]
  strategy: 'ttest'
  n_boot_samples: 200
variance_reduction: 
  method: 'cuped'
  strata: 'country'
  strata_weights:
    'US': 0.5
    'UK': 0.5
result:
  to_csv: True
  csv_path: './result.csv'
```


## Config description 

| parameter | description | values |
| ------------ | ------------ | ------------ | 
| metric_type | Describes metric type | 'solid' for continuous metrics <br /> 'ratio' for ratio metric|
| metric_name | Describes metric name | 'mean', 'median' |
| hypothesis.alpha | Type I error | Must be in [0; 1]. Default is 0.05 |
| hypothesis.beta | Type II error | Must be in [0; 1]. Default is 0.2 |
| hypothesis.alternative | Hypothesis alternative | 'less', 'greater', 'two-sided' |
