from math import ceil
import pandas as pd

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

    def transform(self, X_data):
        
        df_binned = pd.DataFrame({column + '_binned': Binner.equal_pop_binning(X_data[column], *self.num_bins_cut_points_dict[column]) for column in self.column_unique_value_dict})

        return df_binned

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

        # elif method == 'equal_val':



if __name__ == '__main__':
    test_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10, 12, 15, 8, 1, 8, 29, 2, 1, 28, 3]

    binner = Binner()
    # print(binner.find_cut_points(test_list, num_bins=3))
    # print(binner.find_cut_points(test_list, num_bins=4))
    # print(binner.find_cut_points(test_list, num_bins=5))
    # print(binner.__find_cut_points(test_list, num_bins=10))
    print(test_list)
    print(binner.equal_pop_bin(test_list))

