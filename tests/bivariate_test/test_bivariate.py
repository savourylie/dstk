import pandas as pd
import numpy as np
from pandas.testing import assert_frame_equal
from dstk.bivariate_test import cal_information_value

def test_iv():
    df = pd.DataFrame(
        [
            [1, 2, 0, 1],
            [2, 4, 0, 0],
            [3, 6, 0, 0],
            [4, 8, 1, 1],
            [5, 10, 1, 0],
        ],
        columns=["a", "b", "c", "y"])

    iv_df = cal_information_value(df.drop(columns=["y"]), df["y"])
    print(iv_df)

    #assert_frame_equal(psi_df, df_result, check_like=True)
test_iv()