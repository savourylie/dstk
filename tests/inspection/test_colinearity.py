import pandas as pd
import numpy as np
from pandas.testing import assert_frame_equal
from dstk.colinearity_check import cal_VIF

def test_cal_vif():
    df = pd.DataFrame(
        [
            [1, 2, 0],
            [2, 4, 0],
            [3, 6, 0],
            [4, 8, 1],
            [5, 10, 1],
        ],
        columns=["a", "b", "c"])

    vif_df = cal_VIF(df[["a", "b"]])
    print(vif_df)

    #assert_frame_equal(psi_df, df_result, check_like=True)

test_cal_vif()