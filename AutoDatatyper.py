import numpy as np
from tqdm import tqdm

def create_dataset_from_data_column(iterable, label, vector_dim=100, num_rows=5000):
    iterable = np.array(iterable)    
    choice_range = len(iterable)
    iterable_str = iterable.astype(str)
    
    vector_list = []
    for i in tqdm(list(range(num_rows))):
        indices = np.random.choice(choice_range, vector_dim)
        stringified_data = iterable_str[indices]
        
        length_data = np.vectorize(len)(stringified_data)
        sum_data = np.vectorize(lambda x: sum([ord(char) for char in x]))(stringified_data)
        avg_data = sum_data / length_data
        std_data = np.vectorize(lambda x: np.array([ord(char) for char in x]).std())(stringified_data)
        vec = np.concatenate((length_data, sum_data, avg_data, std_data))
        vector_list.append(vec)
        
    return np.array(vector_list), np.array([label] * num_rows)


def get_data_column_type(iterable, estimator, robustness=0.1, vector_dim=100):        
    iterable = np.array(iterable)
    choice_range = len(iterable)
    iterable_str = iterable.astype(str)
        
    vector_list = []
    for i in (range(int(100 * robustness))):
        indices = np.random.choice(choice_range, vector_dim)
        stringified_data = iterable_str[indices]
        
        length_data = np.vectorize(len)(stringified_data)
        sum_data = np.vectorize(lambda x: sum([ord(char) for char in x]))(stringified_data)
        avg_data = sum_data / length_data
        std_data = np.vectorize(lambda x: np.array([ord(char) for char in x]).std())(stringified_data)
        vec = np.concatenate((length_data, sum_data, avg_data, std_data))
        vector_list.append(vec)
    
        prediction = estimator.predict(np.array(vector_list))
        
        decode_dict = {0: 'numeric', 1: 'semantic categorical', 2: 'categorical', 3: 'time'}
        
    return decode_dict[round(prediction.mean())]


def get_data_column_type_df(data, estimator, robustness=0.1, vector_dim=100):
    result_dict = {}
    
    if isinstance(data, pd.DataFrame):
        column_names = data.columns.values
        
        for i, colname in tqdm(enumerate(column_names)):
            datatype = get_data_column_type(data[colname], estimator)
            result_dict[colname] = datatype
    else:
        column_names = list(range(data.shape[1]))
        
        for i, colname in tqdm(enumerate(column_names)):
            datatype = get_data_column_type(data[:, colname], estimator)
            result_dict[colname] = datatype
    
    return result_dict