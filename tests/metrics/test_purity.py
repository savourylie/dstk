from dstk.metrics import purity

def test_purity():
	y_true = [0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1]
	cluster = [1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3]

	assert purity(y_true, cluster) - 0.6666666666666666 < 0.0001 


if __name__ == '__main__':
	test_purity()