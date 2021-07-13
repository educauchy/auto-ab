import numpy as np
import pandas as pd
from typing import Optional


class VarianceReduction:
    def __init__(self, method: str = None):
        self.method = method

    def cuped(self, X: pd.DataFrame, target: str = '', covariate: Optional[str] = None):
        if covariate is None:
            X_corr = X.select_dtypes(include=[np.number]).corr()
            covariate = X_corr.loc[X_corr.index != target, target].sort_values(ascending=False).index[0]
        print('Covariate is column `{}`'.format(covariate))

        cov = X[[target, covariate]].cov().loc[target, covariate]
        var = X[covariate].var()
        self.theta = cov / var
        self.y_cuped = X[target] - self.theta * (X[covariate] - X[covariate].mean())
        X['{}_cuped'.format(target)] = self.y_cuped
        return X

if __name__ == '__main__':
    n = 200000
    t = np.random.randint(160, 195, n)
    df = pd.DataFrame({
        'height': t, # height
        'height_prev': t + np.random.randint(-5, 1, n), # height + random
        'noise': np.random.normal(0, 1, n), # white noise
    })
    df_copy = df.copy(deep=True)

    vr = VarianceReduction()
    ans = vr.cuped(df, target='height', covariate='height_prev')

    print('Means')
    print(np.mean(ans['height']))
    print(np.mean(ans['height_cuped']))
    print('Vars')
    print(np.var(ans['height']))
    print(np.var(ans['height_cuped']))
