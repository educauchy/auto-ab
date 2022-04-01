import pandas as pd
from app.auto_ab.abtest import ABTest


data = pd.read_csv('./storage/data/data.csv')

ab = ABTest(metric_name='mean') # leave default parameters
ab.use_dataset(data, id_col='id', target='height_now', group_col='group')
print( ab.test_hypothesis() )
