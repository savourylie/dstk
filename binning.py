from math import ceil

class Binner(object):
    def __init__(self):
        pass

    def equal_pop_bin(self, arr, num_bins=10):
        num_bins, cut_points = Binner.__find_cut_points(arr, num_bins=num_bins)
        print(num_bins)
        print(cut_points)
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
    def __find_cut_points(arr, method='equal_pop', num_bins=10):
        
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

        elif method == 'equal_val':



if __name__ == '__main__':
    test_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10, 12, 15, 8, 1, 8, 29, 2, 1, 28, 3]

    binner = Binner()
    # print(binner.find_cut_points(test_list, num_bins=3))
    # print(binner.find_cut_points(test_list, num_bins=4))
    # print(binner.find_cut_points(test_list, num_bins=5))
    # print(binner.__find_cut_points(test_list, num_bins=10))
    print(test_list)
    print(binner.equal_pop_bin(test_list))

