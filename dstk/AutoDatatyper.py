import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import Counter

class AutoDatatyper(object):
    def __init__(self, vector_dim=300, num_rows=1000):
        self.vector_dim = vector_dim
        self.num_rows = num_rows
        self.decode_dict = {0: 'numeric', 1: 'character', 2: 'time', 3: 'complex'}
        
    def create_dataset_from_data_column(self, iterable, label):
        iterable_str = self.__remove_na_and_stringify_iterable(iterable)
        choice_range = len(iterable_str)
        
        vector_list = []
        for i in tqdm(list(range(self.num_rows))):
            try:
                vec = self.__get_sample_from_column_data(iterable_str, choice_range)
            except ValueError:
                raise ValueError('All data are NaNs.')
            vector_list.append(vec)

        return np.array(vector_list), np.array([label] * self.num_rows).reshape(-1, 1)

    def __remove_na_and_stringify_iterable(self, iterable):
        # Convert iterable to Series
        if not isinstance(iterable, pd.Series):
            iterable = pd.Series(iterable)
        
        # Drop NAs
        iterable.dropna(inplace=True)        
        iterable = iterable.values
        iterable_str = iterable.astype(str)
        
        return iterable_str
        
    def __get_data_column_type(self, iterable, estimator, robustness):
        iterable_str = self.__remove_na_and_stringify_iterable(iterable)
        choice_range = len(iterable_str)
        
        vector_list = []

        for i in (range(int(100 * robustness))):
            try:
                vec = self.__get_sample_from_column_data(iterable_str, choice_range)
            except ValueError:
                return 'NaN', 1.0
            vector_list.append(vec)

        prediction = estimator.predict(np.array(vector_list))
        prediction_count = Counter(np.vectorize(lambda x: round(x, 1))(prediction))
        confidence = prediction_count.most_common(1)[0][1] / len(prediction)

        return self.decode_dict[round(prediction.mean())], confidence
    
    def get_data_column_type_df(self, data, estimator, robustness=0.1):
        result_dict = {}

        if isinstance(data, pd.DataFrame):
            column_names = data.columns.values

            for i, colname in tqdm(list(enumerate(column_names))):
                datatype, confidence = self.__get_data_column_type(data[colname], estimator, robustness=robustness)
                result_dict[colname] = datatype, confidence
        else:
            column_names = list(range(data.shape[1]))

            for i, colname in tqdm(list(enumerate(column_names))):
                datatype, confidence = self.__get_data_column_type(data[colname], estimator, robustness=robustness)
                result_dict[colname] = datatype, confidence

        return result_dict
    
    def __get_sample_from_column_data(self, iterable_str, choice_range):
        indices = np.random.choice(choice_range, self.vector_dim)
        stringified_data = iterable_str[indices]

        raw_feature_names = ['length', 'max', 'min', 'range', 'sum', 'avg', 'std', 'float', 'time', \
                             'nan', 'json1', 'json2', 'json3', 'array1', 'array2', 'array3', 'array4', \
                            'tag1', 'tag2', 'tag3', 'tag4', 'url']

        raw_feature_dict = {
            'length': np.vectorize(len)(stringified_data),
            'max': np.vectorize(lambda x: max([ord(char) for char in x]))(stringified_data),
            'min': np.vectorize(lambda x: min([ord(char) for char in x]))(stringified_data),
            'range': np.vectorize(lambda x: max([ord(char) for char in x]) - min([ord(char) for char in x]))(stringified_data),
            'sum': np.vectorize(lambda x: sum([ord(char) for char in x]))(stringified_data),
            'avg': np.vectorize(lambda x: sum([ord(char) for char in x]))(stringified_data) / np.vectorize(len)(stringified_data),
            'std': np.vectorize(lambda x: np.array([ord(char) for char in x]).std())(stringified_data),
            'float': np.vectorize(lambda x: x.count('.'))(stringified_data),
            'time': np.vectorize(self.__contains_time_characters)(stringified_data),
            'nan': np.vectorize(self.__is_nan)(stringified_data),
            'json1': np.vectorize(lambda x: x.count('{'))(stringified_data),
            'json2': np.vectorize(lambda x: x.count('}'))(stringified_data),
            'json3': np.vectorize(lambda x: x.count(':'))(stringified_data),
            'array1': np.vectorize(lambda x: x.count('['))(stringified_data),
            'array2': np.vectorize(lambda x: x.count(']'))(stringified_data),
            'array3': np.vectorize(lambda x: x.count(','))(stringified_data),
            'array4': np.vectorize(lambda x: x.count(';'))(stringified_data),
            'tag1': np.vectorize(lambda x: x.count('\\'))(stringified_data),
            'tag2': np.vectorize(lambda x: x.count('/'))(stringified_data),
            'tag3': np.vectorize(lambda x: x.count('|'))(stringified_data),
            'tag4': np.vectorize(lambda x: x.count('-'))(stringified_data),
            'url': np.vectorize(self.__contains_url_characters)(stringified_data)
        }

        range_feature_dict = {feature_name + '_range': 
           np.array([raw_feature_dict[feature_name].max() - raw_feature_dict[feature_name].min()]) for feature_name in raw_feature_names
        }

        sum_feature_dict = {feature_name + '_sum': 
           np.array([raw_feature_dict[feature_name].sum()]) for feature_name in raw_feature_names
        }

        avg_feature_dict = {feature_name + '_avg': 
           np.array([raw_feature_dict[feature_name].mean()]) for feature_name in raw_feature_names
        }

        std_feature_dict = {feature_name + '_std': 
           np.array([raw_feature_dict[feature_name].std()]) for feature_name in raw_feature_names
        }

        count_distinct_feature_dict = {feature_name + '_distinct': 
           np.array([len(Counter(raw_feature_dict[feature_name]))]) for feature_name in raw_feature_names
        }

        concat_list = [value for key, value in raw_feature_dict.items()] \
        + [value for key, value in sum_feature_dict.items()] \
        + [value for key, value in avg_feature_dict.items()] \
        + [value for key, value in std_feature_dict.items()] \
        + [value for key, value in count_distinct_feature_dict.items()]

        vec = np.concatenate(concat_list)

        return vec
    
    def __contains_time_characters(self, string):
            time_chars = {':', '/', '-', '\\', '.', '+',
                         'hr', 'hour', 'min', 'minute', 'sec', 'second',
                         'day', 'week', 'year',
                         '年', '月', '日', '时', '分', '秒',
                         '年', '月', '日', '時', '分', '秒'}

            count = 0
            for char in time_chars:
                if char in string:
                    count += 1
            return count
        
    def __is_nan(self, string):
        return 1 if string.lower() == 'nan' else 0
    
    def __contains_url_characters(self, string):            
            url_chars = {'http', '//', 'www', 'com', 'cn', '_'}
            
            count = 0
            for char in url_chars:
                if char in string:
                    count += 1
            return count
        
    def reduce_data_dict_to_ndarray(self, data_dict):
        return np.concatenate([value[0] for key, value in data_dict.items()], axis=0), np.concatenate([value[1] for key, value in data_dict.items()])

    def consolidate_data(self, foundation_features, new_features, foundation_label, new_label):
        return np.concatenate((foundation_features, new_features), axis=0), np.concatenate((foundation_label, new_label), axis=0) 