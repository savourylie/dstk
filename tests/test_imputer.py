from dstk.preprocessing import Imputer
# from IPython.display import display
import pandas as pd
import numpy as np

def test_mean_imputer():
    df = pd.DataFrame(
        [
        [1, np.nan, np.nan, 1, np.nan], 
        [2, 2, 1, np.nan, np.nan], 
        [5, 1, 2, np.nan, np.nan], 
        [0, 0, 1, 1, 1],
        [np.nan, 0, 2, 13, 1],
        [0, np.nan, 1, 0, 1],
        [0, 0, 1, 1, 1]
        ], 
        columns=["a", "b", "c", "d", "e"])

    df_result = pd.DataFrame(
        [
        [1, 0.59999999999999998, 1.3333333333333333, 1, 1], 
        [2, 2, 1, 3.2000, 1], 
        [5, 1, 2, 3.2000, 1], 
        [0, 0, 1, 1, 1],
        [1.3333333333333333, 0, 2, 13, 1],
        [0, 0.59999999999999998, 1, 0, 1],
        [0, 0, 1, 1, 1]
        ], 
        columns=["a", "b", "c", "d", "e"])

    imputer = Imputer(df.columns) 
    df = imputer.fit_transform(df)

    assert np.isclose(df, df_result).all()

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
    test_mean_imputer()
    test_categorical_imputer()

