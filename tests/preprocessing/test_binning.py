from dstk.feature_binning import FeatureBinning
import pandas as pd
import numpy as np


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
            [1, 1, np.nan],
            [2, 2, np.nan],
            [3, 1, np.nan],
            [4, 0, 1],
            [5, 0, 1],
        ],
        columns=["a", "b", "c"])

    binner = FeatureBinning(df.columns, bin_method="frequency", bin_nums=5)
    df = binner.fit_transform(df)
    print(df)

    #assert df.equals(df_result)

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
    print(df)

    # assert df.equals(df_result)

test_equal_freq_binning()