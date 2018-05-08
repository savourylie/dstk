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

def purity(y_true, cluster):
    compare_df = pd.DataFrame({'y_true': y_true, 'cluster': cluster})

    return sum([sorted(list(compare_df.loc[compare_df['cluster'] == c, 'y_true'].value_counts().items()), key=lambda x: x[1], reverse=True)[0][1] for c in compare_df['cluster'].unique()]) / len(y_true)