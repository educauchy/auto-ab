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

        cov = X[[target, covariate]].cov().loc[target, covariate]
        var = X[covariate].var()
        self.theta = cov / var
        self.y_cuped = X[target] - self.theta * X[covariate]
        X['{}_cuped'.format(target)] = self.y_cuped
        return X

if __name__ == '__main__':
    t = np.random.randint(160, 195, 200)
    df = pd.DataFrame({
        't': t, # height
        'c': t + np.random.randint(0, 100, 200), # height + random
        'f': np.random.normal(0, 1, 200), # white noise
    })
    df_copy = df.copy(deep=True)

    vr = VarianceReduction()
    print(vr.cuped(df, target='t', covariate='c'))
    print(vr.cuped(df_copy, target='t', covariate=None))
