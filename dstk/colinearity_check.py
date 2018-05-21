import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor


def cal_VIF(X):
    variables = X.columns
    vif = pd.DataFrame()
    vif["vif"] = [variance_inflation_factor(X[variables].values, X.columns.get_loc(var)) for var in X.columns]
    vif["feature_names"] = variables
    vif = vif.sort_values(by="vif", ascending=False)
    return vif
