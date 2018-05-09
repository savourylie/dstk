# -*- coding:utf-8 -*-
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class FeatureBinning(BaseEstimator, TransformerMixin):
    '''

    '''

    def __init__(self, feature_list, bin_method='distance', bin_nums=10, bin_labels=None):
        self.bin_method = bin_method
        self.feature_list = feature_list
        self.feature_bins_dict = None
        self.bin_nums = bin_nums
        self.bin_labels=bin_labels
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
                    labels=self.bin_labels
                else:
                    labels=[feature+str(i) for i in range(len(feature_bins)-1)]
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
