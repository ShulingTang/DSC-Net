import numpy as np
from scipy.sparse.linalg import svds
from sklearn.preprocessing import normalize
from sklearn import cluster

def spectual_clustering(C, K, d, alpha):
    # 求sigma(对角矩阵，对角元素为行和，数据类型为m*m)
    kmeansNum = C.shape[1]
    sigma = np.sum(C, axis=1)
    sigma = sigma[0:kmeansNum]
    print('sigma', sigma)
    # 计算sigma的-1/2次方
    sigma = np.diag(sigma**(-0.5))
    print('sigma', sigma)
    # 计算Chat
    C = np.matmul(C, sigma)
    print('Chat', C, '\n', 'Chat.shape', C.shape)



################
    # 该部分还有错误
    r = min(d * K + 1, C.shape[0] - 1)
    U, S, _ = svds(C, r, v0=np.ones(C.shape[0]))
    U = U[:, ::-1]
    S = np.sqrt(S[::-1])
    S = np.diag(S)
    U = U.dot(S)
    U = normalize(U, norm='l2', axis=1)
    Z = U.dot(U.T)
    Z = Z * (Z > 0)
    L = np.abs(Z ** alpha)
    L = L / L.max()
    L = 0.5 * (L + L.T)
    spectral = cluster.SpectralClustering(n_clusters=K, eigen_solver='arpack', affinity='precomputed',
                                          assign_labels='discretize')
    spectral.fit(L)
    grp = spectral.fit_predict(L) + 1
    return grp, L


if __name__ == "__main__":
    C = np.random.randint(1,10,(10,4))
    # 聚类个数
    K = 7
    # 子空间维数
    d = 3
    alpha = 0.4
    y = spectual_clustering(C, K, d, alpha)