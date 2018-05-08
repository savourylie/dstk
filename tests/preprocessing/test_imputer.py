from dstk.preprocessing import Imputer
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

def test_categorical_str_imputer():
    df = pd.DataFrame(
        [['a', 'b', 'a'],
         ['b', 'd', 'c'],
         [np.nan, 'd', np.nan],
         ['a', np.nan, 'c']],
        columns=["a", "b", "c"])

    df_result = pd.DataFrame(
        [['a', 'b', 'a'],
         ['b', 'd', 'c'],
         ['a', 'd', 'c'],
         ['a', 'd', 'c']],
        columns=["a", "b", "c"])

    imputer = Imputer(categorical_features=df.columns, categorical_method="most_freq_class")
    df = imputer.fit_transform(df)

    assert df.equals(df_result)

def test_cont_cat_imputer():
    df = pd.DataFrame(
        {
        'cont1': [1, 1, 1, np.nan, 2, 1, 1],
        'cont2': [2, 2, np.nan, np.nan, -1, -1, -1],
        'cat1': ['a', np.nan, np.nan, np.nan, 'b', 'b', 'b']
        }
        )

    df_result = pd.DataFrame(
        {
        'cont1': [1, 1, 1, 1.1666666666666667, 2, 1, 1],
        'cont2': [2, 2, 0.2, 0.2, -1, -1, -1],
        'cat1': ['a', 'b', 'b', 'b', 'b', 'b', 'b']
        }
        )


    imputer = Imputer(continuous_features=['cont1', 'cont2'], categorical_features=['cat1'], categorical_method="most_freq_class")
    df = imputer.fit_transform(df)

    assert df.equals(df_result)

# def test_cont_cat_median_imputer():
#     df = pd.DataFrame(
#         {
#         'cont1': [1, 1, 1, np.nan, 2, 1, 1],
#         'cont2': [2, 2, np.nan, np.nan, -1, -1, -1],
#         'cat1': ['a', np.nan, np.nan, np.nan, 'b', 'b', 'b']
#         }
#         )

#     df_result = pd.DataFrame(
#         {
#         'cont1': [1, 1, 1, 1, 2, 1, 1],
#         'cont2': [2, 2, -1, -1, -1, -1, -1],
#         'cat1': ['a', 'b', 'b', 'b', 'b', 'b', 'b']
#         }
#         )


#     imputer = Imputer(continuous_features=['cont1', 'cont2'], continuous_method='median', categorical_features=['cat1'], categorical_method="most_freq_class")
#     df = imputer.fit_transform(df)

#     print(df)
#     print(df_result)

#     assert np.isclose(df.values, df_result.values)

# if __name__ == '__main__':
#     test_mean_imputer()
#     test_categorical_imputer()
#     test_categorical_str_imputer()
#     # test_cont_cat_imputer()
