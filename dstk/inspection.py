import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

def equal_dataframes(df1, df2):
    df1 = df1.copy()
    df2 = df2.copy()

    if df1.shape != df2.shape:
        return False

    num_cols = df1.shape[1]

    df1.columns = list(range(num_cols))
    df2.columns = list(range(num_cols))

    numeric_cols = [col for col in df1.columns if is_numeric_dtype(df1[col])]
    str_cols = [col for col in df1.columns if col not in numeric_cols]

    if not np.isclose(df1[numeric_cols], df2[numeric_cols]).all():
        return False

    if not df1[str_cols].equals(df2[str_cols]):
        return False

    return True