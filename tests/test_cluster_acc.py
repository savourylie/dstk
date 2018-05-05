import numpy as np
from dstk.metrics import cluster_acc

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