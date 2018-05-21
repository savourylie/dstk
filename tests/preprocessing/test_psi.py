from dstk.psi import cal_feature_psi
import pandas as pd
from pandas.testing import assert_frame_equal
import numpy as np


def test_psi():
    df1 = pd.DataFrame(
        [
            [1, 1, np.nan],
            [2, 2, np.nan],
            [3, 1, np.nan],
            [4, 0, 1],
            [5, 0, 1],
        ],
        columns=["a", "b", "c"])

    df2 = pd.DataFrame(
        [
            [1, 0, np.nan],
            [2, 0, np.nan],
            [3, 0, np.nan],
            [4, 0, np.nan],
            [5, 0, np.nan],
        ],
        columns=["a", "b", "c"])

    df_result = pd.DataFrame({'feature_names': ["b", "c", "a"], "psi": [np.inf, np.inf, 0]})
    psi_df = cal_feature_psi(df1, df2, bin_nums=5)
    print(psi_df)
    print(df_result)

    assert_frame_equal(psi_df, df_result, check_like=True)


test_psi()
