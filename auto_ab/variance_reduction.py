import numpy as np
import pandas as pd


class VarianceReduction:
    def __init__(self, method: str = None):
        self.method = method

    def cuped(self, X: pd.DataFrame, target: str = '', covariate: str = ''):
        cov = X[[target, covariate]].cov().loc[target, covariate]
        var = X[covariate].var()
        self.theta = cov / var
        self.y_cuped = X[target] - self.theta * X[covariate]
        X['{}_cuped'.format(target)] = self.y_cuped
        return X

if __name__ == '__main__':
    df = pd.DataFrame({
        't': np.random.randint(160, 195, 200), # height
        'c': np.random.randint(60, 120, 200), # weight
    })

    vr = VarianceReduction()
    print(vr.cuped(df, target='t', covariate='c'))