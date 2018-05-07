import numpy as np
import pandas as pd
from dstk.inspection import equal_dataframes


def test_equal_dataframes():
    df1 = pd.DataFrame({'a': [1, 2, 3], 'b': [3, 2, 1], 'c': ['a', 'a', np.nan]})
    df2 = pd.DataFrame({'a': [1.0, 2.0, 3.0], 'b': [3, 2, 1], 'c': ['a', 'a', np.nan]})

    assert equal_dataframes(df1, df2)

def test_equal_dataframes_diff_in_numeric():
    df1 = pd.DataFrame({'a': [1, 2, 4], 'b': [3, 2, 1], 'c': ['a', 'a', np.nan]})
    df2 = pd.DataFrame({'a': [1.0, 2.0, 3.0], 'b': [3, 2, 1], 'c': ['a', 'a', np.nan]})

    assert not equal_dataframes(df1, df2)

def test_equal_dataframes_diff_in_cat():
    df1 = pd.DataFrame({'a': [1, 2, 3], 'b': [3, 2, 1], 'c': ['a', 'a', np.nan]})
    df2 = pd.DataFrame({'a': [1.0, 2.0, 3.0], 'b': [3, 2, 1], 'c': ['b', 'a', np.nan]})

    assert not equal_dataframes(df1, df2)

def test_equal_dataframes_diff_in_colnames():
    df1 = pd.DataFrame({'a': [1, 2, 3], 'b': [3, 2, 1], 'c': ['a', 'a', np.nan]})
    df2 = df1.copy()
    df2.columns = ['d', 'e', 'f']

    assert equal_dataframes(df1, df2)

