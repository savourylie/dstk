from dstk.feature_binning import FeatureBinning
import pandas as pd
import numpy as np
from pandas.testing import assert_frame_equal

def test_equal_freq_binning():
    df = pd.DataFrame(
        [
            [1, 1, np.nan],
            [2, 2, np.nan],
            [3, 1, np.nan],
            [4, 0, 1],
            [5, 0, 1],
        ],
        columns=["a", "b", "c"])

    df_result = pd.DataFrame(
        [
            ["a0", "b1", np.nan],
            ["a1", "b3", np.nan],
            ["a2", "b1", np.nan],
            ["a3", "b0", 1.0],
            ["a4", "b0", 1.0],
        ],
        columns=["a", "b", "c"])

    binner = FeatureBinning(df.columns, bin_method="frequency", bin_nums=5)
    df = binner.fit_transform(df)
    print(df)
    print(df_result)

    assert_frame_equal(df, df_result)


def test_equal_distance_binning():
    df = pd.DataFrame(
        [
            [1, 1, np.nan],
            [2, 2, np.nan],
            [3, 1, np.nan],
            [4, 0, 1],
            [5, 0, 1],
        ],
        columns=["a", "b", "c"])

    binner = FeatureBinning(df.columns, bin_method="distance", bin_nums=5)
    df = binner.fit_transform(df)

    df_result = pd.DataFrame(
        [
            ["a0", "b2", np.nan],
            ["a1", "b4", np.nan],
            ["a2", "b2", np.nan],
            ["a3", "b0", 1.0],
            ["a4", "b0", 1.0],
        ],
        columns=["a", "b", "c"])

    print(df)
    print(df_result)
    assert_frame_equal(df, df_result)

test_equal_freq_binning()
test_equal_distance_binning()