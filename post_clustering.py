import numpy as np
from scipy.sparse.linalg import svds
from sklearn.preprocessing import normalize
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, adjusted_mutual_info_score
from sklearn.cluster import KMeans

nmi = normalized_mutual_info_score
ami = adjusted_mutual_info_score
ari = adjusted_rand_score


def acc(y_true, y_pred):
    """
    Calculate clustering accuracy.
    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`
    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    # from sklearn.utils.linear_assignment_ import linear_assignment
    from scipy.optimize import linear_sum_assignment as linear_assignment
    ind_row, ind_col = linear_assignment(w.max() - w)
    return sum([w[i, j] for i, j in zip(ind_row, ind_col)]) * 1.0 / y_pred.size


def err_rate(gt_s, s):
    return 1.0 - acc(gt_s, s)

def post_proC(C, K, d, alpha):
    # 求sigma(对角矩阵，对角元素为行和，数据类型为m*m)
    # C.shape = (n,m)
    n = C.shape[0]
    kmeansNum = C.shape[1]
    sigma = np.sum(C, axis=1)
    sigma = sigma[0:kmeansNum]
    # print('sigma', sigma)
    # 计算sigma的-1/2次方
    sigma = np.diag(sigma**(-0.5))
    C = np.matmul(C, sigma)
    # print('Chat', C, '\n', 'Chat.shape', C.shape)
    r = min(d * K + 1, C.shape[1] - 1)
    U, Sigma, _ = svds(C, r, v0=np.ones(kmeansNum))
    U = normalize(U, norm='l2', axis=1)
    y = KMeans(n_clusters=K, random_state=0).fit(U)
    y = y.labels_
    return y, U

def spectral_clustering(C, K, d, alpha, ro):
    y, _ = post_proC(C, K, d, ro)
    return y



def matlab_max(A, B):
    shape = A.shape
    a = A.ravel()
    b = B.ravel()
    c = []
    for i in range(a.size):
        if a[i]>b[i]:
            ci = a[i]
        else:ci = b[i]
        c = np.r_[c, ci]
    c = c.reshape(shape)
    return c

def mysvd(C, ReducedDim=None):
    #  You can change this number according your machine computational power
    max_matrix_size = 1600
    eigvector_ratio = 0.1

    if not ReducedDim is None:
        k = 0
    nSmp, mFea = C.shape
    if mFea/nSmp > 1.0713:
        ddata = np.matmul(C, C.T)
        ddata = matlab_max(ddata, ddata.T)

        dimMatric = ddata.shape(1)
        if ReducedDim > 0 and dimMatric > max_matrix_size and ReducedDim < dimMatric*eigvector_ratio:






