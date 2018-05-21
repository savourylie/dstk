from dstk.transformation import FeatureBinning, Binner
import pandas as pd
import numpy as np

def test_binning_ck():
    # test_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10, 12, 15, 8, 1, 8, 29, 2, 1, 28, 3]
    col1 = [2 6 7 9 2 7 4 2]
    col2 = [6 9 4 1 1 5 5 9]
    col3 = [5 8 3 0 9 8 6 4]

    df = pd.DataFrame({'col1': col1, 'col2': col2, 'col3': col3})
    df_result = pd.DataFrame({})
    binner = Binner(num_bins=2)
    # print(binner.find_cut_points(test_list, num_bins=3))
    # print(binner.find_cut_points(test_list, num_bins=4))
    # print(binner.find_cut_points(test_list, num_bins=5))
    # print(binner.__find_cut_points(test_list, num_bins=10))
    binner.fit(df)

    print(binner.num_bins_cut_points_dict)
    print(binner.column_unique_value_dict)
    print(binner.columns_binned)

    print(binner.transform(test_df))