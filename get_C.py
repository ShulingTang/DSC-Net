from sklearn.cluster import KMeans
import numpy as np
from cvxopt import matrix, solvers
import torch
import matlab.engine
def get_Coefficient(x, kmeansNum):
    """
    In this part, first conduct kmeans to obtain M anchor points,
    then construct a coefficient matrix C through the anchor points
    and all the data.
    :param x: sampled data
    :param kmeansNum:anchor graph
    :return C:Coefficient Matrix
    """
    # x.shape = (n, d), m.shape = (m, d)
    alpha = 1.0
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    # x = x.numpy()
    m = KMeans(n_clusters=kmeansNum, random_state=0).fit(x)
    num, dim = x.shape
    m = m.cluster_centers_   # m.shape = (m, d)
    '''
    # 该部分为翻译的matlab代码
    h = 2*alpha*np.identity(kmeansNum)+2*np.matmul(m, m.T)
    h = matrix(1/2*(h+h.T))  # h.shape = (m, m)
    bb = x.T   # B.shape = (d, n)
    z = []
    l = matrix(np.ones(kmeansNum))

    # I = matrix(I.T)
    o = matrix(0.0)
    for i in range(num):
        # fi.shape = (m ,1)
        fi = np.matmul(-2*(bb[:, i]).T, m.T)
        # fi.type = ndarray　shape = {tuple}(m,)
        fi = fi.astype(np.float64)
        fi = matrix(fi.reshape(1, kmeansNum))
        fi = matrix(fi)
        zi = solvers.qp(h, fi.T, G=None, h=None, A=l.T, b=o, kktsolver=None)
        # print(zi.shape)
        zi = np.array(zi['x']).reshape(kmeansNum)
        z = np.r_[z, zi]
    z = z.reshape(num, kmeansNum)  # z.shape = (n, m)
    z = z.astype(np.float32)
    z = torch.from_numpy(z).to('cuda')
    m = torch.from_numpy(m).to('cuda')
    '''
    ############
    # 这部分直接调用matlab中的代码来获取参数C
    a, b = x.shape
    c, d = m.shape
    x = x.tolist()
    mm = m.tolist()
    eng = matlab.engine.start_matlab()
    z = eng.get_c(x, mm, alpha, a, b, c, d)
    eng.quit()
    # print(type(z))
    z = np.array(z)
    z = z.astype(np.float32)
    z = torch.from_numpy(z).to('cuda')
    # z = z.T
    m = torch.from_numpy(m).to('cuda')
    return z, m


if __name__ == "__main__":
    C = np.random.randint(1, 10, (1000, 216))
    # 聚类个数
    K = 200
    # 子空间维数

    alpha = 1
    y = get_Coefficient(C, K)
