from sklearn.cluster import KMeans
import numpy as np
from cvxopt import matrix, solvers

def get_Coefficient(X, kmeansNum, alpha):
    """
    In this part, first conduct kmeans to obtain M anchor points,
    and then construct a coefficient matrix C through the anchor points
    and all the data.
    :param X: sampled data
    :param M:anchor graph
    :return C:Coefficient Matrix
    """
    # X.shape = (n, d), M.shape = (m, d)
    M = KMeans(n_clusters=kmeansNum, random_state=0).fit(X)
    num, dim = X.shape
    M = M.cluster_centers_
    A = 2*alpha*np.identity(kmeansNum)+2*np.matmul(M, M.T)
    A = matrix(1/2*(A+A.T))  # A.shape = (m, m)
    B = X.T  # B.shape = (d, n)
    Z = []
    I = matrix(np.ones(kmeansNum))
    # I = matrix(I.T)
    print(I.shape)
    b = matrix(0.0)
    for i in range(num):
        # fi.shape = (m ,1)
        fi = np.matmul(-2*(B[:, i]).T, M.T)
        fi = matrix(fi.reshape(1, kmeansNum))
        zi = solvers.qp(A, fi, G=None, h=None, A=I, b=b, kktsolver=None)
        Z = np.r_[Z, zi]
    return Z


if __name__ == "__main__":
    C = np.random.randint(1, 10, (1000, 216))
    # 聚类个数
    K = 200
    # 子空间维数

    alpha = 1
    y = get_Coefficient(C, K, alpha)