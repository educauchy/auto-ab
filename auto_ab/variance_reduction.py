import numpy as np
import pandas as pd
from typing import Optional, List
import statsmodels.api as sm


class VarianceReduction:
    def __init__(self, method: str = None):
        self.method = method

    def _predict_target(self, X: pd.DataFrame, target_prev: str = '',
                       factors_prev: List[str] = None, factors_now: List[str] = None) -> pd.Series:
        """
        Simple linear regression for covariate prediction
        :param X: Pandas DataFrame
        :param target_prev: Target on previous period column name
        :param factors_prev: Factor columns for modelling
        :param factors_now: Factor columns for prediction on current period
        :return: Pandas Series with predicted values
        """
        Y = X[target_prev]
        X_train = X[factors_prev]
        model = sm.OLS(Y, X_train)
        results = model.fit()
        print(results.summary())
        X_predict = X[factors_now]
        return results.predict(X_predict)

    def cupac(self, X: pd.DataFrame, target_prev: str = '', target_now: str = '',
              factors_prev: List[str] = None, factors_now: List[str] = None, groups: str = '') -> pd.DataFrame:
        """
        Perform CUPAC with prediction of target column on experiment period.
        Original paper: https://doordash.engineering/2020/06/08/improving-experimental-power-through-control-using-predictions-as-covariate-cupac/.
        Previous period = before experiment, now_period = after experiment.
        :param X: Pandas DataFrame for analysis
        :param target_prev: Target on previous period column name
        :param target_now: Target on current period column name
        :param factors_prev: Factor columns for modelling
        :param factors_now: Factor columns for prediction on current period
        :param groups: Groups A and B column name
        :return: Pandas DataFrame with additional columns: target_pred and target_now_cuped
        """
        X.loc[:, 'target_pred'] = self._predict_target(X, target_prev, factors_prev, factors_now)
        X_new = self.cuped(X, target_now, groups, 'target_pred')
        return X_new

    def cuped(self, X: pd.DataFrame, target: str = '', groups: str = '',
              covariate: Optional[str] = None) -> pd.DataFrame:
        """
        Perform CUPED on target column with known/unknown covariate.
        Original paper: https://exp-platform.com/Documents/2013-02-CUPED-ImprovingSensitivityOfControlledExperiments.pdf.
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

        for group in X[groups].unique():
            X_subdf = X[X[groups] == group]
            group_y_cuped = X_subdf[target] - self.theta * (X_subdf[covariate] - X_subdf[covariate].mean())
            X.loc[X[groups] == group, '{}_cuped'.format(target)] = group_y_cuped
        return X

if __name__ == '__main__':
    df = pd.read_csv('../data/internal/guide/ab_data.csv')

    vr = VarianceReduction()
    ans = vr.cupac(df, target_prev='height_prev', target_now='height_now',
                   factors_prev=['weight_prev'],
                   factors_now=['weight_now'], groups='groups')
    # ans = vr.cuped(df, target='height_now', groups='groups', covariate='height_prev')

    target_var = 'height_now'
    target_cuped = target_var + '_cuped'
    print('\nMeans')
    print(np.mean(ans[target_var]))
    print(np.mean(ans.loc[ans.groups == 'A', target_var]))
    print(np.mean(ans.loc[ans.groups == 'B', target_var]))
    print(np.mean(ans[target_cuped]))
    print(np.mean(ans.loc[ans.groups == 'A', target_cuped]))
    print(np.mean(ans.loc[ans.groups == 'B', target_cuped]))
    print('\nVars')
    print(np.var(ans[target_var]))
    print(np.var(ans.loc[ans.groups == 'A', target_var]))
    print(np.var(ans.loc[ans.groups == 'B', target_var]))
    print(np.var(ans[target_cuped]))
    print(np.var(ans.loc[ans.groups == 'A', target_cuped]))
    print(np.var(ans.loc[ans.groups == 'B', target_cuped]))
