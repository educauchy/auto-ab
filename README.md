# auto_ab - automate your A/B-testing

**auto_ab** is a library developed to automate routine tasks during runs of multiple online controlled experiments, also known as A/B-tests.

Current features:
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

Parameters for A/B-test are set in config file. The example of config file you can find below.

```yaml
metric:
  metric_type: 'ratio'
data:
  id_col: 'id'
  target: 'height'
  numerator: 'clicks'
  denominator: 'sessions'
simulation:
  n_iter: 100
  split_rates: [0.1, 0.2, 0.3, 0.4, 0.5]
  increment:
    vars: [0, 1, 2, 3, 4, 5]
    extra_params: []
hypothesis:
  alpha: 0.05
  beta: 0.8
  alternative: 'two-sided'
  strategy: ''
  strata: 'country'
  strata_weights:
    'US': 0.5
    'UK': 0.5
  n_boot_samples: 200
result:
  to_csv: True
  csv_path: './result.csv'
splitter:
  split_rate: 0.5
```
