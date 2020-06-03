import numpy as np


def spectual_clustering(A):

    print('A', A)
    print('A.shape', A.shape)
    D = np.sum(A, axis=1)
    print('D', D)
    D = D[0:3]
    # D =D.reshape(A.shape[0])
    print('D', D)
    D = np.diag(D)
    print('D', D)

    return D


if __name__ == "__main__":
    A = np.arange(1, 13).reshape(4, 3)
    spectual_clustering(A)