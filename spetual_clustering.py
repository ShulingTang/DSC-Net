import numpy as np
from scipy.sparse.linalg import svds
from sklearn.preprocessing import normalize
from sklearn import cluster
from numpy import linalg as la
from sklearn.cluster import KMeans

def spectual_clustering(C, K, d, alpha):
    # 求sigma(对角矩阵，对角元素为行和，数据类型为m*m)
    kmeansNum = C.shape[1]
    sigma = np.sum(C, axis=1)
    sigma = sigma[0:kmeansNum]
    print('sigma', sigma)
    # 计算sigma的-1/2次方
    sigma = np.diag(sigma**(-0.5))
    print('sigma', sigma)
    # 计算用于svd的矩阵Ｃ＝Ｃ×ｓｉｇｍａ
    C = np.matmul(C, sigma)
    print('Chat', C, '\n', 'Chat.shape', C.shape)

    U, Sigma, VT = la.svd(C)
    print('U:',U)
    print('\nSigma:', Sigma)
    print('\nVT:', VT)
    y = KMeans(n_clusters=K, random_state=0).fit(U)
    return y


    # U, S, Vh = svds(C)


    '''
    # 该部分还有错误
    r = min(d * K + 1, C.shape[1] - 1)
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
    '''

if __name__ == "__main__":
    C = np.random.randint(1,10,(100,40))
    # 聚类个数
    K = 2
    # 子空间维数
    d = 3
    alpha = 0.4
    y = spectual_clustering(C, K, d, alpha)