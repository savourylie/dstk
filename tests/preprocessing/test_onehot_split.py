import numpy as np
import pandas as pd
from dstk.preprocessing import onehot_split

def test_onehot_split():
    df = pd.DataFrame({'cat': [1, 1, 0, 0, 1, 1, np.nan], 'cont': [2, 1, 0.1, 2, 0.99, 0.21, np.nan]})

    df_result1 = pd.DataFrame({'cont': [2, 1, 0.1, 2, 0.99, 0.21, np.nan]})
    df_result2 = pd.DataFrame({'cat': [1, 1, 0, 0, 1, 1, np.nan]})

    df1, df2 = onehot_split(df)
    assert df1.equals(df_result1)
    assert df2.equals(df_result2)