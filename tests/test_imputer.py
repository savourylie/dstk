from dstk.feature_preprocessing import Imputer
from IPython.display import display
import pandas as pd
import numpy as np


def test_median_imputer():
    df = pd.DataFrame(
        [[1, 2, 2, 3, 4, 1, 1, 1, np.nan], 
        [1, 1, 20, 2, 2, 2, 1, np.nan, np.nan], 
        [1, 5, 5, 5, 5, 1, 2, np.nan, np.nan], 
        [2, 1, 0, 0, 0, 0, 1, 1, 1]], 
        columns=["a", "b", "c", "d", "e", "f", "g", "h", "i"])

    df_result = pd.DataFrame(
        [[1, 2, 2, 3, 4, 1, 1, 1, 0.5], 
        [1, 1, 20, 2, 2, 2, 1, 2, 2], 
        [1, 5, 5, 5, 5, 1, 2, 5, 5], 
        [2, 1, 0, 0, 0, 0, 1, 1, 1]], 
        columns=["a", "b", "c", "d", "e", "f", "g", "h", "i"])

    imputer = Imputer(df.columns) 
    df = imputer.fit_transform(df)

    assert df_result.equals(df)

if __name__ == '__main__':
    test_median_imputer()



