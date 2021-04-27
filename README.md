# auto-ab

The library is developed to assess AB-tests and provide power analysis.

For a custom research you need to load dataset with a particular structure shown below.
Also, you need to provide output variable type in order to use appropriate statistical test.

split_variable | output | confound | timestamp
---------- | ---------- | ---------- | ---------- 
control | 1 | Europe | 14092
treatment | 1 | Asia | 23094
control | 0 | America | 32810
treatment | 1 | Asia | 17823
control | 1 | Asia | 9482
control | 0 | Europe | 22944
treatment | 0 | America | 31091

where
* *split_variable* indicates groups for experiment;
* *output* contains metric of the experiment;
* *confound* is a confounding variable, by which data must be divided into equal size groups;
* *timestamp* is a timestamp of the action.

If timestamp is not presented in data, then data is considered in samples order.
