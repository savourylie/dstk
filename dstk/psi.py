import pandas as pd
import numpy as np


def cal_feature_psi(expected_data, actual_data=None, feature_list=[], ratio=0.5, bin_nums=10):
    if actual_data is None:
        half_len = int(ratio * len(expected_data))
        expected_data = expected_data.iloc[:half_len, :]
        actual_data = expected_data.iloc[half_len:, :]
    feature_psi_list = list()
    if len(feature_list) == 0:
        feature_list = list(expected_data.columns)
    for feature in feature_list:
        try:
            print(feature)
            psi = cal_psi(expected_data[feature].astype(float), actual_data[feature].astype(float), bin_nums=bin_nums,
                          is_print=False)
        except Exception as e:
            print(feature, e)
            continue
        feature_psi_list.append({'feature_names': feature, 'psi': psi})
    feature_psi_df = pd.DataFrame(feature_psi_list)
    feature_psi_df = feature_psi_df.sort_values(by='psi', ascending=False).reset_index(drop=True)
    return feature_psi_df


def cal_psi(expected_data, actual_data, bin_nums=10, is_print=True):
    """
    This funcion is used to calculate population stability index of a same data in two time periods

    :param actual_data:
    :param expected_data:
    :param bin_nums:
    :return:
    """
    # 计算
    _, bin_bounds = pd.cut(expected_data, bins=bin_nums, retbins=True, include_lowest=True)
    # print(bin_bounds)
    actual_cut = (pd.cut(actual_data, bins=np.unique(bin_bounds)).value_counts()) / len(
        actual_data)
    expected_cut = (pd.cut(expected_data, bins=np.unique(bin_bounds)).value_counts()) / len(
        expected_data)
    # print actual_cut, expected_cut
    actual_pct = actual_cut.sort_index()
    expected_pct = expected_cut.sort_index()
    actual_pct_copy = list(actual_pct)
    expected_pct_copy = list(expected_pct)
    psi = sum(
        [(actual_pct[i] - expected_pct_copy[i]) * np.log(actual_pct_copy[i] / expected_pct_copy[i]) if expected_pct_copy[i] != 0 else 0 for
         i in range(len(actual_pct))])
    if is_print:
        results = pd.DataFrame({'expected_pct': expected_pct.values,
                                'actual_pct': actual_pct.values,
                                'diff': (actual_pct.values - expected_pct.values) * np.log(
                                    actual_pct / expected_pct)},
                               index=expected_pct.index)
        cols = ['expected_pct', 'actual_pct', 'diff']
        results = results[cols]
        print(results)
        # return {'data': results, 'statistic': psi}
        # return results
    return psi
