from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np


class Imputer(BaseEstimator, TransformerMixin):
    def __init__(self, continuous_features, continuous_method, categorical_features=None, categorical_method=None,
                 drop_threshold=0.95):
        """

        :param continuous_features: 连续型变量list
        :param continuous_method: 连续型变量缺失值处理方法，包括mean, turncated_mean, median, bin_nan
        :param categorical_features: 离散型变量list
        :param categorical_method: 离散型变量缺失值处理方法, freq_class
        :param drop_threshold:
        """
        self.continuous_features = continuous_features
        self.continuous_method = continuous_method
        self.categorical_features = categorical_features
        self.categorical_method = categorical_method
        self.drop_threshold = drop_threshold

        self.invalid_features = None
        self.continuous_pad_vals = None
        self.categorical_pad_vals = None

    def fit(self, X):
        # step 1: delete features which missing rate are larger than drop threshold
        nan_rate = pd.DataFrame(X.isnull().sum() / len(X)).reset_index().rename(
            columns={"index": "feature_name", 0: "nan_rate"})
        invalid_features = list(nan_rate[nan_rate["nan_rate"].astype(float) >= self.drop_threshold]["feature_name"])
        self.invalid_features = invalid_features

        # step 2: update feature columns
        self.categorical_features = [x for x in self.categorical_features if x not in invalid_features]
        self.continuous_features = [x for x in self.continuous_features if x not in invalid_features]

        # step 3: 对两种特征分开处理
        self.deal_categorical(X, self.categorical_features)
        self.deal_continuous(X, self.continuous_features)

        return self

    def transform(self, X):
        X = self.deal_categorical(X, self.categorical_features)
        X = self.deal_continuous(X, self.continuous_features)
        return X

    def fit_transform(self, X, y=None, **fit_params):
        return self.fit(X).transform(X)

    def deal_continuous(self, X, continuous_features):

        # 针对已经fit过后的transform
        if self.continuous_pad_vals is not None:
            X[continuous_features] = X[continuous_features].fillna(self.continuous_pad_vals)
        # 重新fit
        else:
            if self.continuous_method == "mean":
                self.continuous_pad_vals = X[continuous_features].mean()
            elif self.continuous_method == "truncated_mean":
                X_truncated = X[X <= X.quantile(0.95)]
                self.continuous_pad_vals = X_truncated[continuous_features].mean()
            elif self.continuous_method == "median":
                self.continuous_pad_vals = X[continuous_features].median()
            # TODO!!!!!
            elif self.continuous_method == "bin_nan":
                pass
            else:
                X[self.continuous_features] = X[self.continuous_features].fillna(0)
        return X

    def deal_categorical(self, X, categorical_features):
        if self.categorical_pad_vals is not None:
            X[categorical_features] = X[categorical_features].fillna(self.categorical_pad_vals)
        else:
            if self.categorical_method == "freq_class":
                self.categorical_pad_vals = [X[c].value_counts().index[0] for c in categorical_features]
            else:
                X[self.categorical_features] = X[self.categorical_features].fillna(0)
        return X


if __name__ == '__main__':
    df = pd.DataFrame([[1, 2, np.nan], [3, np.nan, np.nan], [1, np.nan, np.nan], [2, 1, 0]], columns=["a", "b", "c"])
    # nan_df = pd.DataFrame(df.isnull().sum() / len(df)).reset_index().rename(
    #     columns={"index": "feature_name", 0: "nan_rate"})
    # print(nan_df)
    print(df["a"].value_counts())
    # df[["a", "b"]]=df[["a", "b"]].fillna(df[["a", "b"]].mean())
    # print(df[["a", "b"]].mean())
