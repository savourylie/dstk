from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np


class Imputer(BaseEstimator, TransformerMixin):
    def __init__(self, continuous_features=None, continuous_method='mean', categorical_features=None,
                 categorical_method='most_freq_class',
                 drop_threshold=0.95):
        """

        :param continuous_features: column names of continuous features (list)
        :param continuous_method: imputing methods for continuous features ['mean', 'turncated_mean', 'median', 'bin_nan'] (str)
        :param categorical_features: column names of categorical features (list)
        :param categorical_method: imputing methods for categorical features ['most_freq_class', 'stringify']
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
        # step 1: delete features of which missing rates are larger than the drop threshold
        nan_rate = pd.DataFrame(X.isnull().sum() / len(X)).reset_index().rename(
            columns={"index": "feature_name", 0: "nan_rate"})
        invalid_features = list(nan_rate[nan_rate["nan_rate"].astype(float) >= self.drop_threshold]["feature_name"])
        self.invalid_features = invalid_features

        # step 2: update feature columns
        ### LDC TODO: exception handling for where categorical_features is None
        if self.continuous_features is not None:
            self.continuous_features = [x for x in self.continuous_features if x not in invalid_features]
            # step 3: deal with two types of features separately
            _ = self.fill_continuous(X, self.continuous_features)

        if self.categorical_features is not None:
            self.categorical_features = [x for x in self.categorical_features if x not in invalid_features]
            # step 3: deal with two types of features separately
            _ = self.fill_categorical(X, self.categorical_features)

        if self.continuous_features is None and self.categorical_features is None:
            raise ValueError("continuous features and categorical features cannot both be None.")

        return self

    def transform(self, X):
        if self.continuous_features is not None:
            X = self.fill_continuous(X, self.continuous_features)
        if self.categorical_features is not None:
            X = self.fill_categorical(X, self.categorical_features)
        return X

    def fit_transform(self, X, y=None, **fit_params):
        return self.fit(X).transform(X)

    def fill_continuous(self, X, continuous_features):
        # 针对已经fit过后的transform
        if self.continuous_pad_vals is not None:
            X[continuous_features] = X[continuous_features].fillna(self.continuous_pad_vals)
        # 重新fit
        else:
            if self.continuous_method == "mean":
                self.continuous_pad_vals = X[continuous_features].mean()
            elif self.continuous_method == "truncated_mean":
                X_truncated = X[(X <= X.quantile(0.95)) & (X >= X.quantile(0.05))]
                self.continuous_pad_vals = X_truncated[continuous_features].mean()
            elif self.continuous_method == "median":
                self.continuous_pad_vals = X[continuous_features].median()
            # TODO!!!!! Bin-nan imputing method
            elif self.continuous_method == "bin_nan":
                pass
            else:
                raise ValueError("Imputing method (continuous) is invalid.")
                # X[self.continuous_features] = X[self.continuous_features].fillna(0)
        return X

    def fill_categorical(self, X, categorical_features):
        if self.categorical_pad_vals is not None:
            X[categorical_features] = X[categorical_features].fillna(self.categorical_pad_vals)
        else:
            if self.categorical_method == "most_freq_class":
                self.categorical_pad_vals = {c: X[c].value_counts().index[0] for c in categorical_features}
            elif self.categorical_method == "stringify":
                self.categorical_pad_vals = "None"
            else:
                raise ValueError("Imputing method (categorical) is invalid.")
                # X[self.categorical_features] = X[self.categorical_features].fillna(0)
        return X


if __name__ == '__main__':
    df = pd.DataFrame([[1, 2, np.nan], [3, np.nan, np.nan], [1, np.nan, np.nan], [2, 1, 0]], columns=["a", "b", "c"])
    print(df["a"].value_counts())
