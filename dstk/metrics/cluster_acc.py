import numpy as np
import pandas as pd
from collections import Counter
from IPython.display import display

def cluster_acc(y_true, cluster):
    compare_df = pd.DataFrame({'y_true': y_true, 'cluster': cluster})

    classes = set(y_true)
    cluster_codes = [x for x, y in sorted(Counter(cluster).items(), key=lambda x: x[1], reverse=True)]
    cluster_class_acc_dict = dict()

    # display(compare_df)

    for i, code in enumerate(cluster_codes):
        # Get counts for each class, by cluster code
        temp_class_dict = compare_df.loc[compare_df['cluster'] == code, 'y_true'].value_counts().to_dict()
        # Get total number of data
        temp_num_total_data = sum(temp_class_dict.values())
        # Update dictionary with available classes
        temp_class_dict = {key: temp_class_dict[key] for key in classes if key in temp_class_dict}
        # Get majority class for the cluster code
        try:
            temp_class = max(temp_class_dict, key=temp_class_dict.get)
        except ValueError:
            temp_num_bad = temp_num_total_data
            temp_class = None
        else:
            # Get number of wrongly clustered data
            temp_num_bad = temp_num_total_data - temp_class_dict[temp_class]
        # Record number of wrongly clustered data for the cluster
        cluster_class_acc_dict[code] = temp_num_bad
        # Update the remaining class set
        if temp_class is not None:
            classes.remove(temp_class)

    return 1 - (sum(cluster_class_acc_dict.values()) / len(compare_df))

def test_cluster_acc():
    # Test1
    y_true1 = [0, 0, 0, 1, 1, 1, 1, 1, 1]
    y_pred1 = [1, 1, 1, 0, 0, 0, 0, 0, 0]
    # # Test 2
    y_true2 = [0, 0, 1, 1, 1]
    y_pred2 = [1, 1, 1, 0, 0]
    # Test 3
    y_true3 = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 2, 0]
    y_pred3 = [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2]
    # Test 4
    y_true4 = [1, 1, 1, 0, 0, 0, 2, 2, 2, 1]
    y_pred4 = [1, 1, 1, 2, 2, 1, 2, 0, 0, 1]
    assert np.abs(cluster_acc(y_true1, y_pred1) - 1) < 0.0001
    assert np.abs(cluster_acc(y_true2, y_pred2) - 0.8) < 0.0001
    assert np.abs(cluster_acc(y_true3, y_pred3) - 0.7368421052631579) < 0.0001
    assert np.abs(cluster_acc(y_true4, y_pred4) - 0.8) < 0.0001

if __name__ == '__main__':
    test_cluster_acc()
