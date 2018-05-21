import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from math import ceil


class FeatureBinning(BaseEstimator, TransformerMixin):
    '''

    '''
    def __init__(self, feature_list, bin_method='distance', bin_nums=10, bin_labels=None):
        self.bin_method = bin_method
        self.feature_list = feature_list
        self.feature_bins_dict = None
        self.bin_nums = bin_nums
        self.bin_labels = bin_labels
        # self.is_set_feature_names = False
        # self.feature_names = list()

    def fit(self, X, y=None, **kwargs):
        if self.bin_method == 'distance':
            self.feature_bins_dict = self.binning_by_distance(self.feature_list, X)
        elif self.bin_method == 'frequency':
            self.feature_bins_dict = self.binning_by_frequency(self.feature_list, X)
        else:
            raise ValueError("Binning method is invalid.")
        # self.is_set_feature_names = True
        return self

    def transform(self, X):
        if self.feature_bins_dict is not None:
            # print('用train上的结果分桶')
            for feature in list(self.feature_bins_dict.keys()):
                feature_bins = self.feature_bins_dict[feature]
                if self.bin_labels is not None:
                    labels = self.bin_labels
                else:
                    labels = [feature+str(i) for i in range(len(feature_bins) - 1)]
                X[feature] = pd.cut(X[feature], feature_bins, include_lowest=True, labels=labels)
        # if self.is_set_feature_names is True:
        #     self.feature_names = list(X.columns)
        # self.is_set_feature_names = False
        return X

    def fit_transform(self, X, y=None, **fit_params):
        return self.fit(X).transform(X)

    def binning_by_distance(self, feature_list, feature_df):
        """
        特征的等距分桶：每一bin距离相同的标准为特征分桶

        :param feature_list: 需要分桶的特征
        :param feature_df: 原始数据的dataframe
        :param bin_nums: 分桶的数量
        :return: 特征：[分桶边界] dict
        """
        copy_df = feature_df.copy()
        feature_bins_dict = {}
        for feature in feature_list:
            unique_vals = feature_df[feature].nunique()
            if unique_vals <= 2:
                # 如果已经是二值化的类别类特征就不需要再做离散化
                print('{}已经是二值化特征了，不需做分桶'.format(feature))
                continue
            else:
                # 特征的分桶区间边界
                try:
                    _, bins = pd.cut(copy_df[feature], bins=self.bin_nums, retbins=True)
                    feature_bins_dict[feature] = list(bins)
                except Exception as e:
                    print(feature, e)
        return feature_bins_dict

    def binning_by_frequency(self, feature_list, feature_df):
        """
        特征的等频分桶，分桶边界为特征的分位点值

        :param feature_list: 需要分桶的特征
        :param feature_df: 原始数据的dataframe
        :param bin_nums:
        :return: 特征：[分桶边界] dict
        """
        copy_df = feature_df.copy()
        feature_quantiles_dict = {}
        for feature in feature_list:
            num_of_unique = feature_df[feature].nunique()
            if num_of_unique <= 2:
                print('{}已经是二值化特征，不需要分桶'.format(feature))
                continue
            else:
                try:
                    _, bins = pd.qcut(copy_df[feature], q=self.bin_nums, retbins=True, duplicates="drop")
                    feature_quantiles_dict[feature] = list(bins)
                except Exception as e:
                    print(feature, e)
        return feature_quantiles_dict


class Binner(object):
    def __init__(self, method='equal_pop', num_bins=10, ignore_factor=2):
        self.method = method
        self.num_bins = num_bins
        self.ignore_factor = ignore_factor
        self.num_bins_cut_points_dict = {}
        self.column_unique_value_dict = {}
        self.columns = None
        self.columns_binned = None

    def fit(self, X_data):
        if self.columns is not None:
            data = X_data.loc[:, self.columns].copy()
        else:
            self.columns = X_data.columns
            data = X_data.copy()

        self.num_bins_cut_points_dict = {column: Binner.find_cut_points(X_data[column], method=self.method, num_bins=self.num_bins) for column in self.columns}
        self.column_unique_value_dict = {column: len(X_data[column].value_counts()) for column in self.columns if len(X_data[column].value_counts()) > self.num_bins * self.ignore_factor}
        self.columns_binned = self.column_unique_value_dict.keys()

        return self

    def transform(self, X_data):
        
        df_binned = pd.DataFrame({column + '_binned': Binner.equal_pop_binning(X_data[column], *self.num_bins_cut_points_dict[column]) for column in self.column_unique_value_dict})

        return df_binned

    def fit_transform(self, X_data):

        return self.fit(X_data).transform(X_data)

    @staticmethod
    def equal_pop_binning(arr, num_bins, cut_points):
        arr_binned = []

        for i, x in enumerate(arr):
            assigned = False

            for j, y in enumerate(cut_points):
                high, low = y
                if low < x <= high:
                    arr_binned.append(j)
                    assigned = True

                else:
                    continue

            if assigned is not True:
                if x > cut_points[0][0]:
                    arr_binned.append(0)

                elif x < cut_points[-1][-1]:
                    arr_binned.append(len(cut_points) - 1)

                else:
                    raise ValueError("Code is wrong!")

        return arr_binned



    @staticmethod
    def find_cut_points(arr, method, num_bins=10):
        
        len_arr = len(arr)
        bin_size = ceil(len_arr / num_bins)

        if method == 'equal_pop':
            arr_sorted = sorted(list(arr), reverse=True)
            cut_points = []
            prev_pos = 0
            pos = bin_size

            for i, x in enumerate(arr_sorted):
                if i < pos:
                    if pos < len_arr:
                        continue

                    else:
                        cut_points.append((arr_sorted[prev_pos], min(arr_sorted) - 1))

                        num_bins = len(cut_points)
                        return num_bins, cut_points

                if arr_sorted[pos] == arr_sorted[pos - 1]:
                    if i == len_arr - 1:
                        cut_points.append((arr_sorted[prev_pos], arr_sorted[pos] - 1))

                        num_bins = len(cut_points)
                        return num_bins, cut_points

                    else:
                        pos += 1

                else:
                    if i == len_arr - 1:
                        cut_points.append((arr_sorted[prev_pos], arr_sorted[pos] - 1))

                        num_bins = len(cut_points)
                        return num_bins, cut_points
                        
                    else:
                        cut_points.append((arr_sorted[prev_pos], arr_sorted[pos]))
                    prev_pos = pos
                    pos += bin_size