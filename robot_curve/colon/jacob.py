import os
import torch
import torch.nn as nn

import warnings

warnings.filterwarnings('ignore')

from include import *
from PIL import Image
import PIL

import numpy as np
import torch
import torch.optim
from torch.autograd import Variable

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
dtype = torch.cuda.FloatTensor
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
print("num GPUs", torch.cuda.device_count())


def pad_circular1d(x, pad):
    x = torch.cat([x, x[0:pad]])
    x = torch.cat([x[-2 * pad:-pad], x])
    return x


class Pad1d(torch.nn.Module):
    def __init__(self, pad):
        super(Pad1d, self).__init__()
        self.pad = pad

    def forward(self, x):
        shape = [1, x.shape[1], x.shape[2] + 2 * self.pad]
        xx = Variable(torch.zeros(shape)).type(dtype)
        for i in range(x.shape[1]):
            xx[0, i] = pad_circular1d(x[0, i], self.pad)
        return xx  # pad_circular1d(x, self.pad)


def print_filters(net):
    for m in net.modules():
        if isinstance(m, nn.Conv1d):
            print(m.weight.data.cpu().numpy())


def plot_gradients(out_grads):
    for i, g in enumerate(out_grads):
        plt.semilogy(g, label=str(i))
    plt.legend()
    plt.show()


def conv(in_f, out_f, kernel_size, stride=1, bias=False, pad=True):
    '''
    Circular convolution
    '''
    to_pad = int((kernel_size - 1) / 2)
    if pad:
        padder = Pad1d(to_pad)
    else:
        padder = None

    # convolver = nn.utils.weight_norm(nn.Conv1d(in_f, out_f, kernel_size, stride, padding=0, bias=bias), name='weight')
    convolver = nn.Conv1d(in_f, out_f, kernel_size, stride, padding=0, bias=bias)

    layers = filter(lambda x: x is not None, [padder, convolver])
    return nn.Sequential(*layers).type(dtype)


def initialize(net):
    for m in net.modules():
        if isinstance(m, nn.Conv1d):
            if m.weight.data.shape[2] > 1:
                m.weight.data[0, 0, 0] = 0
                m.weight.data[0, 0, 1] = 1
                m.weight.data[0, 0, 2] = 1
                m.weight.data[0, 0, 3] = 0
                m.weight.data[0, 0, 4] = 0


def decnet(
        num_output_channels=1,
        num_channels_up=[1] * 5,
        filter_size_up=2,
        act_fun=nn.ReLU(),  # nn.LeakyReLU(0.2, inplace=True)
        modeout="none",
        upsample_mode='none',
        scale_factor=2,
):
    num_channels_up = num_channels_up + [num_channels_up[-1]]
    n_scales = len(num_channels_up)

    model = nn.Sequential()

    for i in range(len(num_channels_up) - 1):

        if upsample_mode != 'none':
            model.add(nn.Upsample(scale_factor=scale_factor, mode=upsample_mode))

        model.add(conv(num_channels_up[i], num_channels_up[i + 1], filter_size_up, 1))
        if act_fun != None:
            model.add(act_fun)
        # model.add(ChannelNormalization(num_channels_up[i],mode=mode))
        model.add(nn.BatchNorm1d(num_channels_up[i], affine=True))

    if modeout != "none":
        model.add(nn.BatchNorm1d(num_channels_up[i + 1], affine=True))
    # model.add(conv( num_channels_up[-1], num_output_channels, 1,bias=True,pad=False))

    model.add(conv(num_channels_up[-1], num_output_channels, 1, bias=False, pad=False))
    model[-1][0].weight.requires_grad = False
    a = int(num_channels_up[-1] / 2)
    w = np.concatenate((np.ones(a), -np.ones(a))) / np.sqrt(2 * a)
    model[-1][0].weight.data = torch.tensor(w[None, :, None])

    # model.add(nn.Sigmoid())

    return model


def get_ni(num_channels, n, upsamplefactor):
    length = int(n / upsamplefactor)
    shape = [1, num_channels[0], length]
    print("input shape : ", shape)
    net_input = Variable(torch.zeros(shape)).type(dtype)
    net_input.data.uniform_()
    return net_input


def fit(net,
        y,
        y_clean,
        num_channels,
        net_input=None,
        num_iter=5000,
        LR=0.01,
        upsamplefactor=1,
        ):
    if net_input is not None:
        print("input provided")
    else:
        print("new input generated")
        net_input = get_ni(num_channels, y.data.shape[2], upsamplefactor)

    net_input = net_input.type(dtype)
    net_input_saved = net_input.data.clone()
    noise = net_input.data.clone()
    p = [x for x in net.parameters()]

    mse_wrt_noisy = np.zeros(num_iter)
    mse_wrt_truth = np.zeros(num_iter)

    # optimizer = torch.optim.SGD(p, lr=LR,momentum=0.9)
    optimizer = torch.optim.Adam(p, lr=LR)

    mse = torch.nn.MSELoss()  # .type(dtype)

    nconvnets = 0
    for p in list(filter(lambda p: len(p.data.shape) > 2, net.parameters())):
        nconvnets += 1

    out_grads = np.zeros((nconvnets, num_iter))
    out_filters = np.zeros((nconvnets + 1, num_iter))

    for i in range(num_iter):
        def closure():
            optimizer.zero_grad()
            out = net(net_input.type(dtype))

            # training loss 
            loss = mse(out, y)
            loss.backward()
            mse_wrt_noisy[i] = loss.data.cpu().numpy()

            # actual loss
            true_loss = mse(Variable(out.data, requires_grad=False).type(dtype), y_clean.type(dtype))
            mse_wrt_truth[i] = true_loss.data.cpu().numpy()

            # output gradients
            for ind, p in enumerate(
                    list(filter(lambda p: p.grad is not None and len(p.data.shape) > 2, net.parameters()))):
                out_grads[ind, i] = p.grad.data.norm(2).item()

            # output norms of filters
            ind = 0
            for m in net.modules():
                if isinstance(m, nn.Conv1d):
                    out_filters[ind, i] = m.weight.data.norm(2).item()
                    ind += 1
                    # print(m.weight.data.cpu().numpy())

            if i % 10 == 0:
                print('Iteration %05d    Train loss %f' % (i, loss.data), '\r', end='')

            return loss

        loss = optimizer.step(closure)

    return mse_wrt_noisy, mse_wrt_truth, net_input, net, out_grads, out_filters


def get_jacobian(net, x):
    y = net(x)
    noutputs = y.shape[2]
    outimgs = torch.eye(noutputs)
    jac = []
    for outimg in outimgs:
        y = net(x)
        y.backward(outimg[None, None, :].type(dtype))
        allgrads = []
        for ind, p in enumerate(list(
                filter(lambda p: p.grad is not None and len(p.data.shape) > 2 and p.data.shape[0] > 1,
                       net.parameters()))):
            gra = p.grad.data.cpu().numpy()
            gra = gra.flatten()
            allgrads += [gra]
        totalgra = np.concatenate((allgrads))
        jac += [totalgra]
    return np.array(jac)


## Generate  noise and step function to be fitted
n = 512
# Gaussian noise
shape = [1, 1, n]
noise_np = np.random.normal(scale=0.5, size=shape)
noise = Variable(torch.from_numpy(noise_np)).type(dtype)

# step function
ystep_np = np.zeros([1, n])
ystep_np[0, :int(n / 2)] = np.ones(int(n / 2))
ystep = np_to_var(ystep_np).type(dtype)
ystep_np = ystep_np[0]

# noisy copy
y = ystep + 0.4 * noise
y_np = y.data.cpu().numpy()

print(f"ystep: {ystep}, noisy_y: {y}")
