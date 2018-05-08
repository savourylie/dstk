# -*- coding:utf-8 -*-
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from .feature_utils import binning_by_distance, binning_by_frequency


class FeatureBinning(BaseEstimator, TransformerMixin):
    '''

    '''
    def __init__(self, feature_list, bin_method='distance'):
        self.bin_method = bin_method
        self.feature_list = feature_list
        self.feature_bins_dict = None
        #self.is_set_feature_names = False
        #self.feature_names = list()

    def fit(self, X, y=None, **kwargs):
        if self.bin_method == 'distance':
            self.feature_bins_dict = binning_by_distance(self.feature_list, X)
        if self.bin_method == 'frequency':
            self.feature_bins_dict = binning_by_frequency(self.feature_list, X)
        #self.is_set_feature_names = True
        return self

    def transform(self, X):
        if self.feature_bins_dict is not None:
            #print('用train上的结果分桶')
            for feature in self.feature_list:
                info_dict = self.feature_bins_dict[feature]
                # 各桶的边界
                feature_bins = info_dict['bins']
                # 每桶对应的值
                bin_values = info_dict['vals']
                X[feature] = pd.cut(X[feature], feature_bins, labels=bin_values)
        # if self.is_set_feature_names is True:
        #     self.feature_names = list(X.columns)
        # self.is_set_feature_names = False
        return X

    def fit_transform(self, X, y=None, **fit_params):
        return self.fit(X, y).transform(X)

    def binning_by_distance(self, feature_list, feature_df, bin_nums=10):
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
            num_of_unique_vals = feature_df[feature].unique().shape[0]
            #print('num of unique values in', feature, 'is:', num_of_unique_vals)
            if num_of_unique_vals <= 2:
                # 如果已经是二值化的类别类特征就不需要再做离散化
                print('已经是二值化特征了，不需做分桶')
                continue
            else:
                # 特征的分桶区间边界
                _, bins = pd.cut(copy_df[feature], bins=bin_nums, retbins=True)
                #  feature_bins = list(np.linspace(feature_min, feature_max, (bin_nums + 1)))
                feature_bins_dict[feature] = list(bins)
                return feature_bins_dict

    def binning_by_frequency(self, feature_list, feature_df, bin_nums=10):
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
            num_of_unique = feature_df[feature].unique()[0]
            if num_of_unique <= 2:
                print('已经是二值化特征，不需要分桶')
                continue
            else:
                _, bins = pd.qcut(copy_df[feature], q=bin_nums, retbins=True)
                feature_quantiles_dict[feature] = list(bins)
        return feature_quantiles_dict