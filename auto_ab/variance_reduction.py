import numpy as np
import pandas as pd
from typing import Optional


class VarianceReduction:
    def __init__(self, method: str = None):
        self.method = method

    def cuped(self, X: pd.DataFrame, target: str = '', groups: str = '', covariate: Optional[str] = None) -> pd.DataFrame:
        """
        Perform CUPED on target column with known/unknown covariate
        :param X: Pandas DataFrame for analysis
        :param target: Target column name
        :param groups: Groups A and B column name
        :param covariate: Covariate column name. If None, then most correlated column in considered as covariate
        :return: Pandas DataFrame with additional target CUPEDed column
        """
        if covariate is None:
            X_corr = X.select_dtypes(include=[np.number]).corr()
            covariate = X_corr.loc[X_corr.index != target, target].sort_values(ascending=False).index[0]
        print('Covariate is column `{}`'.format(covariate))

        cov = X[[target, covariate]].cov().loc[target, covariate]
        var = X[covariate].var()
        self.theta = cov / var

        for group in ['A', 'B']:
            X_subdf = X[X[groups] == group]
            group_y_cuped = X_subdf[target] - self.theta * (X_subdf[covariate] - X_subdf[covariate].mean())
            X.loc[X[groups] == group, '{}_cuped'.format(target)] = group_y_cuped
        return X

if __name__ == '__main__':
    n = 200000
    t = np.random.randint(160, 195, n)
    df = pd.DataFrame({
        'height': t, # height
        'height_prev': t + np.random.randint(-5, 1, n), # height + random
        'noise': np.random.normal(0, 1, n), # white noise
        'groups': np.random.choice(['A', 'B'], size=n)
    })
    df_copy = df.copy(deep=True)

    vr = VarianceReduction()
    ans = vr.cuped(df, target='height', groups='groups', covariate='height_prev')
    print(ans)

    print('Means')
    print(np.mean(ans['height']))
    print(np.mean(ans.loc[ans.groups == 'A', 'height']))
    print(np.mean(ans.loc[ans.groups == 'B', 'height']))
    print(np.mean(ans['height_cuped']))
    print(np.mean(ans.loc[ans.groups == 'A', 'height_cuped']))
    print(np.mean(ans.loc[ans.groups == 'B', 'height_cuped']))
    print('Vars')
    print(np.var(ans['height']))
    print(np.var(ans.loc[ans.groups == 'A', 'height']))
    print(np.var(ans.loc[ans.groups == 'B', 'height']))
    print(np.var(ans['height_cuped']))
    print(np.var(ans.loc[ans.groups == 'A', 'height_cuped']))
    print(np.var(ans.loc[ans.groups == 'B', 'height_cuped']))
