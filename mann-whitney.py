from scipy.stats import mannwhitneyu
from numpy.random import normal
import matplotlib.pyplot as plt


alpha = 0.05

d1 = normal(4, 2, 150)
d2 = normal(4, 1.8, 150)

plt.hist(d1)
plt.hist(d2)
plt.show()

result = mannwhitneyu(d1, d2, alternative='two-sided')
print(result)
print('Statistic: {}'.format(result[0]))
print('P-value: {}'.format(round(result[1], 5)))

if result[1] < alpha:
    print('Different median')
else:
    print('Same median')

