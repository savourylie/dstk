from dstk.feature_preprocessing import Imputer
# from IPython.display import display
import pandas as pd
import numpy as np


def test_categorical_imputer():
    df = pd.DataFrame(
        [[1, 2, 2, 8],
         [1, 1, np.nan, 3],
         [np.nan, np.nan, np.nan, np.nan],
         [2, 1, np.nan, -1]],
        columns=["a", "b", "c", "d"])

    df_result=pd.DataFrame(
        [[1, 2, 2, 8],
         [1, 1, 2, 3],
         [1, 1, 2, -1],
         [2, 1, 2, -1]],
        columns=["a", "b", "c", "d"])

    imputer = Imputer(categorical_features=df.columns, categorical_method="most_freq_class")
    df = imputer.fit_transform(df)

    assert np.isclose(df, df_result).all()


if __name__ == '__main__':
    test_categorical_imputer()
