"""
By Xifeng Guo (guoxifeng1990@163.com), May 13, 2020.
All rights reserved.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from post_clustering_pre import spectral_clustering, acc, nmi
import scipy.io as sio
import math
from sklearn.cluster import KMeans
from get_C import get_Coefficient


class Conv2dSamePad(nn.Module):
    """
    Implement Tensorflow's 'SAME' padding mode in Conv2d.
    When an odd number, say `m`, of pixels are need to pad, Tensorflow will pad one more column at right or one more
    row at bottom. But Pytorch will pad `m+1` pixels, i.e., Pytorch always pads in both sides.
    So we can pad the tensor in the way of Tensorflow before call the Conv2d module.
    """

    def __init__(self, kernel_size, stride):
        super(Conv2dSamePad, self).__init__()
        self.kernel_size = kernel_size if type(kernel_size) in [list, tuple] else [kernel_size, kernel_size]
        self.stride = stride if type(stride) in [list, tuple] else [stride, stride]

    def forward(self, x):
        # print('shape', x.shape)
        # in_height = x.size(2)
        # in_width = x.size(3)

        in_height = x.shape[2]
        in_width = x.shape[3]
        out_height = math.ceil(float(in_height) / float(self.stride[0]))
        out_width = math.ceil(float(in_width) / float(self.stride[1]))
        pad_along_height = ((out_height - 1) * self.stride[0] + self.kernel_size[0] - in_height)
        pad_along_width = ((out_width - 1) * self.stride[1] + self.kernel_size[1] - in_width)
        pad_top = math.floor(pad_along_height / 2)
        pad_left = math.floor(pad_along_width / 2)
        pad_bottom = pad_along_height - pad_top
        pad_right = pad_along_width - pad_left
        return F.pad(x, [pad_left, pad_right, pad_top, pad_bottom], 'constant', 0)


class ConvTranspose2dSamePad(nn.Module):
    """
    This module implements the "SAME" padding mode for ConvTranspose2d as in Tensorflow.
    A tensor with width w_in, feed it to ConvTranspose2d(ci, co, kernel, stride), the width of output tensor T_nopad:
        w_nopad = (w_in - 1) * stride + kernel
    If we use padding, i.e., ConvTranspose2d(ci, co, kernel, stride, padding, output_padding), the width of T_pad:
        w_pad = (w_in - 1) * stride + kernel - (2*padding - output_padding) = w_nopad - (2*padding - output_padding)
    Yes, in ConvTranspose2d, more padding, the resulting tensor is smaller, i.e., the padding is actually deleting row/col.
    If `pad`=(2*padding - output_padding) is odd, Pytorch deletes more columns in the left, i.e., the first ceil(pad/2) and
    last `pad - ceil(pad/2)` columns of T_nopad are deleted to get T_pad.
    In contrast, Tensorflow deletes more columns in the right, i.e., the first floor(pad/2) and last `pad - floor(pad/2)`
    columns are deleted.
    For the height, Pytorch deletes more rows at top, while Tensorflow at bottom.
    In practice, we usually want `w_pad = w_in * stride`, i.e., the "SAME" padding mode in Tensorflow,
    so the number of columns to delete:
        pad = 2*padding - output_padding = kernel - stride
    We can solve the above equation and get:
        padding = ceil((kernel - stride)/2), and
        output_padding = 2*padding - (kernel - stride) which is either 1 or 0.
    But to get the same result with Tensorflow, we should delete values by ourselves instead of using padding and
    output_padding in ConvTranspose2d.
    To get there, we check the following conditions:
    If pad = kernel - stride is even, we can directly set padding=pad/2 and output_padding=0 in ConvTranspose2d.
    If pad = kernel - stride is odd, we can use ConvTranspose2d to get T_nopad, and then delete `pad` rows/columns by
    ourselves; or we can use ConvTranspose2d to delete `pad - 1` by setting `padding=(pad - 1) / 2` and `ouput_padding=0`
    and then delete the last row/column of the resulting tensor by ourselves.
    Here we implement the former case.
    This module should be called after the ConvTranspose2d module with shared kernel_size and stride values.
    And this module can only output a tensor with shape `stride * size_input`.
    A more flexible module can be found in `yaleb.py` which can output arbitrary size as specified.
    """

    def __init__(self, kernel_size, stride):
        super(ConvTranspose2dSamePad, self).__init__()
        self.kernel_size = kernel_size if type(kernel_size) in [list, tuple] else [kernel_size, kernel_size]
        self.stride = stride if type(stride) in [list, tuple] else [stride, stride]

    def forward(self, x):
        in_height = x.size(2)
        in_width = x.size(3)
        pad_height = self.kernel_size[0] - self.stride[0]
        pad_width = self.kernel_size[1] - self.stride[1]
        pad_top = pad_height // 2
        pad_bottom = pad_height - pad_top
        pad_left = pad_width // 2
        pad_right = pad_width - pad_left
        return x[:, :, pad_top:in_height - pad_bottom, pad_left: in_width - pad_right]


class ConvAE(nn.Module):
    def __init__(self, channels, kernels):
        """
        :param channels: a list containing all channels including the input image channel (1 for gray, 3 for RGB)
        :param kernels:  a list containing all kernel sizes, it should satisfy: len(kernels) = len(channels) - 1.
        """
        super(ConvAE, self).__init__()
        assert isinstance(channels, list) and isinstance(kernels, list)
        self.encoder = nn.Sequential()
        for i in range(1, len(channels)):
            #  Each layer will divide the size of feature map by 2
            self.encoder.add_module('pad%d' % i, Conv2dSamePad(kernels[i - 1], 2))
            self.encoder.add_module('conv%d' % i,
                                    nn.Conv2d(channels[i - 1], channels[i], kernel_size=kernels[i - 1], stride=2))
            self.encoder.add_module('relu%d' % i, nn.ReLU(True))

        self.decoder = nn.Sequential()
        channels = list(reversed(channels))
        kernels = list(reversed(kernels))
        for i in range(len(channels) - 1):
            # Each layer will double the size of feature map
            self.decoder.add_module('deconv%d' % (i + 1),
                                    nn.ConvTranspose2d(channels[i], channels[i + 1], kernel_size=kernels[i], stride=2))
            self.decoder.add_module('padd%d' % i, ConvTranspose2dSamePad(kernels[i], 2))
            self.decoder.add_module('relud%d' % i, nn.ReLU(True))

    def forward(self, x):
        h = self.encoder(x)
        y = self.decoder(h)
        return y


class SelfExpression(nn.Module):
    def __init__(self, c):
        super(SelfExpression, self).__init__()
        # torch.nn.Paramater 将一个不可训练的类型Tensor转换成可以训练的类型parameter并将这个parameter绑定到这个module里面
        self.Coefficient = nn.Parameter(c, requires_grad=True)
        # self.land = False
    def forward(self, x):  # shape=[n, d]
        # if not self.land:
        #     c, m = get_Coefficient(x, kmeansNum)
        #     self.Coefficient = nn.Parameter(c)
        #     self.m = nn.Parameter(m)
        #     self.land = True
        # x = torch.from_numpy(x).to('cuda')
        y = torch.matmul(self.Coefficient, x)
        return y

class kmeans(nn.Module):
    # 初始化这里不增加这个x会报错：需要输入一个参数，但是输入了两个，并不知道道哪两个....
    def __init__(self, x):
        super(kmeans, self).__init__()
        # torch.nn.Paramater 将一个不可训练的类型Tensor转换成可以训练的类型parameter并将这个parameter绑定到这个module里面
        # self.Coefficient = nn.Parameter(1.0e-8 * torch.ones(n, m, dtype=torch.float32), requires_grad=True)

    def forward(self, x):  # shape=[n, d]
        x = x.detach().cpu().numpy()
        y = KMeans(n_clusters=kmeansNum, random_state=0).fit(x)
        y = y.cluster_centers_
        y = torch.from_numpy(y).to('cuda')
        return y

class DSCNet(nn.Module):
    def __init__(self, channels, kernels, num_sample, c):
        super(DSCNet, self).__init__()
        self.n = num_sample
        # self.kmeansNum = kmeansNum
        self.ae = ConvAE(channels, kernels)
        self.self_expression = SelfExpression(c)
        # self.land = land
        # self.K_means = kmeans(self.kmeansNum)

    def forward(self, x, m):  # shape=[n, c, w, h]
        z = self.ae.encoder(x)
        # self expression layer, reshape to vectors, multiply Coefficient, then reshape back
        shape = z.shape
        # print("z.shape is:", shape)
        z = z.view(self.n, -1)  # shape=[n, d]
        # M = self.K_means(z)
        z_recon= self.self_expression(m)  # shape=[n, d]
        # z_recon = torch.from_numpy(z_recon).to('cuda')
        z_recon_reshape = z_recon.view(shape)
        x_recon = self.ae.decoder(z_recon_reshape)  # shape=[n, c, w, h]
        return x_recon, z, z_recon

    def loss_fn(self, x, x_recon, z, z_recon, weight_coef, weight_selfExp):
        # print("x.shape:", x.shape)
        # print("x_recon.shape:", x_recon.shape)
        # print("z.shape:", z.shape)
        # print("z_recon.shape:", z_recon.shape)
        loss_ae = F.mse_loss(x_recon, x, reduction='sum')
        # print('loss_ae:', loss_ae)
        loss_coef = torch.sum(torch.pow(self.self_expression.Coefficient, 2))
        # print('loss_coef', loss_coef)
        # F.mse_loss均方误差损失
        loss_selfExp = F.mse_loss(z_recon, z, reduction='sum')
        # print('loss_selfExp:', loss_selfExp)
        loss = loss_ae + weight_coef * loss_coef + weight_selfExp * loss_selfExp

        return loss


def train(model,  # type: DSCNet
          x, y, epochs, lr=1e-3, weight_coef=1.0, weight_selfExp=150, device='cuda',
          alpha=0.04, dim_subspace=12, ro=8, show=5):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # print('model.parameters():', model.parameters())
    # 判断一个对象是不是一个已和类型（这里为判断是不是张量）
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, dtype=torch.float32, device=device)
    x = x.to(device)
    if isinstance(y, torch.Tensor):
        y = y.to('cpu').numpy()
    # np.unique该函数是去除数组中的重复数字,并进行排序之后输出.
    K = len(np.unique(y))
    for epoch in range(epochs):
        # land = model.self_expression.land
        x_recon, z, z_recon = model(x, m)
        loss = model.loss_fn(x, x_recon, z, z_recon, weight_coef=weight_coef, weight_selfExp=weight_selfExp)
        # zero the parameter gradients（该部分可以理解为随机梯度下降）
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % show == 0 or epoch == epochs - 1:
            C = model.self_expression.Coefficient.detach().to('cpu').numpy()
            print('C:', C)
            # M = model.
            y_pred = spectral_clustering(C, K, dim_subspace, alpha, ro)
            print('Epoch %02d: loss=%.4f, acc=%.4f, nmi=%.4f' %
                  (epoch, loss.item() / y_pred.shape[0], acc(y, y_pred), nmi(y, y_pred)))


if __name__ == "__main__":
    import argparse
    import warnings
    import time


    start = time.time()
    # ArgumentParser参数解析器，描述它做了什么
    parser = argparse.ArgumentParser(description='DSCNet')
    # add_argument函数来增加参数
    parser.add_argument('--db', default='orl',
                        choices=['coil20', 'coil100', 'orl', 'reuters10k', 'stl', 'mnist'])
    parser.add_argument('--show-freq', default=10, type=int)
    parser.add_argument('--ae-weights', default=None)
    parser.add_argument('--save-dir', default='results')
    # parse_args获取解析的参数
    args = parser.parse_args()
    print(args)
    import os

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    db = args.db
    if db == 'coil20':
        # load data
        data = sio.loadmat('datasets/COIL20.mat')
        x, y = data['fea'].reshape((-1, 1, 32, 32)), data['gnd']

        # np.squeeze(x)从数组的形状中删除单维条目，即把shape中为1的维度去掉
        y = np.squeeze(y - 1)  # y in [0, 1, ..., K-1]

        # network and optimization parameters
        num_sample = x.shape[0]
        channels = [1, 15]
        kernels = [3]
        epochs = 40
        weight_coef = 1.0
        weight_selfExp = 75
        kmeansNum = 200

        # post clustering parameters
        alpha = 0.04  # threshold of C
        dim_subspace = 12  # dimension of each subspace
        ro = 8  #
        warnings.warn("You can uncomment line#64 in post_clustering.py to get better result for this dataset!")
    elif db == 'coil100':
        # load data
        data = sio.loadmat('datasets/COIL100.mat')
        x, y = data['fea'].reshape((-1, 1, 32, 32)), data['gnd']
        y = np.squeeze(y - 1)  # y in [0, 1, ..., K-1]

        # network and optimization parameters
        num_sample = x.shape[0]
        channels = [1, 50]
        kernels = [5]
        epochs = 120
        weight_coef = 1.0
        weight_selfExp = 15
        kmeansNum = 800
        # post clustering parameters
        alpha = 0.04  # threshold of C
        dim_subspace = 12  # dimension of each subspace
        ro = 8  #
    elif db == 'orl':
        # load data
        data = sio.loadmat('datasets/ORL_32x32.mat')
        x, y = data['fea'].reshape((-1, 1, 32, 32)), data['gnd']
        y = np.squeeze(y - 1)  # y in [0, 1, ..., K-1]
        # network and optimization parameters
        num_sample = x.shape[0]
        channels = [1, 3, 3, 5]
        kernels = [3, 3, 3]
        epochs = 700
        weight_coef = 2.0
        weight_selfExp = 0.2
        kmeansNum = 50

        # post clustering parameters
        alpha = 0.2  # threshold of C
        dim_subspace = 3  # dimension of each subspace
        ro = 1  #
    elif db == 'mnist':
        # load data
        data = sio.loadmat('datasets/training_data.mat')
        x, y = data['X'].reshape((-1, 1, 28, 28)), data['y']
        I = np.identity(10)
        y = np.squeeze(y - 1)  # y in [0, 1, ..., K-1]
        # y = I[:, y]

        # network and optimization parameters
        num_sample = x.shape[0]
        channels = [1, 3, 3, 5]
        kernels = [3, 3, 3]
        epochs = 700
        weight_coef = 2.0
        weight_selfExp = 0.2
        kmeansNum = 100

        # post clustering parameters
        alpha = 0.2  # threshold of C
        dim_subspace = 3  # dimension of each subspace
        ro = 1  #

    '''
        z = self.ae.encoder(x)
        # self expression layer, reshape to vectors, multiply Coefficient, then reshape back
        shape = z.shape
        # print("z.shape is:", shape)
        z = z.view(self.n, -1)  # shape=[n, d]
        # M = self.K_means(z)
        z_recon= self.self_expression(c, m)  # shape=[n, d]
    '''
    x = torch.tensor(x, dtype=torch.float32)
    zz = ConvAE(channels, kernels).encoder(x)
    zz = zz.view(num_sample, -1)
    c, m = get_Coefficient(zz, kmeansNum)
    print('c1:', c)
    # m = torch.tensor(m, dtype=torch.float32)
    dscnet = DSCNet(num_sample=num_sample, channels=channels, kernels=kernels, c=c)
    dscnet.to(device)

    # load the pretrained weights which are provided by the original author in
    # https://github.com/panji1990/Deep-subspace-clustering-networks
    ae_state_dict = torch.load('pretrained_weights_original/%s.pkl' % db)
    dscnet.ae.load_state_dict(ae_state_dict)
    print("Pretrained ae weights are loaded successfully.")

    train(dscnet, x, y, epochs, weight_coef=weight_coef, weight_selfExp=weight_selfExp,
          alpha=alpha, dim_subspace=dim_subspace, ro=ro, show=args.show_freq, device=device)
    end = time.time()
    print(str(time))
    torch.save(dscnet.state_dict(), args.save_dir + '/%s-model.ckp' % args.db)
