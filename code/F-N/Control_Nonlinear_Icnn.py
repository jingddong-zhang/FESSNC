import numpy as np
from numpy.core.defchararray import mod
import torch
import torch.nn as nn
import torch.nn.functional as F
from spectral_normalization import SpectralNorm
from MonotonicNN import MonotonicNN
import math
class ICNN(nn.Module):
    def __init__(self, input_shape, layer_sizes, activation_fn):
        super(ICNN, self).__init__()
        self._input_shape = input_shape
        self._layer_sizes = layer_sizes
        self._activation_fn = activation_fn
        ws = []
        bs = []
        us = []
        prev_layer = input_shape
        w = torch.empty(layer_sizes[0], *input_shape)
        nn.init.xavier_normal_(w)
        ws.append(nn.Parameter(w))
        b = torch.empty([layer_sizes[0], 1])
        nn.init.xavier_normal_(b)
        bs.append(nn.Parameter(b))
        for i in range(len(layer_sizes))[1:]:
            w = torch.empty(layer_sizes[i], *input_shape)
            nn.init.xavier_normal_(w)
            ws.append(nn.Parameter(w))
            b = torch.empty([layer_sizes[i], 1])
            nn.init.xavier_normal_(b)
            bs.append(nn.Parameter(b))
            u = torch.empty([layer_sizes[i], layer_sizes[i-1]])
            nn.init.xavier_normal_(u)
            us.append(nn.Parameter(u))
        self._ws = nn.ParameterList(ws)
        self._bs = nn.ParameterList(bs)
        self._us = nn.ParameterList(us)

    def forward(self, x):
        # x: [batch, data]
        if len(x.shape) < 2:
            x = x.unsqueeze(0)
        else:
            data_dims = list(range(1, len(self._input_shape) + 1))
            x = x.permute(*data_dims, 0)
        z = self._activation_fn(torch.addmm(self._bs[0], self._ws[0], x))
        for i in range(len(self._us)):
            u = F.softplus(self._us[i])
            w = self._ws[i + 1]
            b = self._bs[i + 1]
            z = self._activation_fn(torch.addmm(b, w, x) + torch.mm(u, z))
        return z

class ControlNet(torch.nn.Module):
    
    def __init__(self,n_input,n_hidden,n_output):
        super(ControlNet, self).__init__()
        torch.manual_seed(2)
        self.layer1 = SpectralNorm(torch.nn.Linear(n_input, n_hidden))
        self.layer2 = SpectralNorm(torch.nn.Linear(n_hidden,n_hidden))
        self.layer3 = SpectralNorm(torch.nn.Linear(n_hidden,n_output))

    def forward(self,x):
        sigmoid = torch.nn.ReLU()
        h_1 = sigmoid(self.layer1(x))
        h_2 = sigmoid(self.layer2(h_1))
        out = self.layer3(h_2)
        return out

def ZBF(x):
    M = 5.
    # return M**2-torch.sum(x[:,0:2]**2,dim=1)
    max,_ =  torch.max(x**2,dim=1)
    return M**2-max


class LyapunovFunction(nn.Module):
    def __init__(self,n_input,n_hidden,n_output,input_shape,smooth_relu_thresh=0.1,layer_sizes=[64, 64],eps=1e-3):
        super(LyapunovFunction, self).__init__()
        torch.manual_seed(2)
        self._d = smooth_relu_thresh
        self._icnn = ICNN(input_shape, layer_sizes, self.smooth_relu)
        self._eps = eps
        self._control = ControlNet(n_input,n_hidden,n_output)
        self._mono = MonotonicNN(1, [10,10], nb_steps=50)
   

    def forward(self, x):
        g = self._icnn(x)
        g0 = self._icnn(torch.zeros_like(x))
        u = self._control(x)
        u0 = self._control(torch.zeros_like(x))
        m_h = self._mono(ZBF(x).unsqueeze(1))[:, 0]
        return self.smooth_relu(g - g0) + self._eps * x.pow(2).sum(dim=1), u*x , m_h
        # return self.smooth_relu(g - g0) + self._eps * x.pow(2).sum(dim=1), u-u0 

    def derivative(self,x):
        F,u=self.forward(x)
        dF=torch.autograd.grad(F.sum(),x,create_graph=True)[0]
        # dF = torch.autograd.functional.jacobian(self.forward, data)
        solenoidal_field=dF
        # HF = torch.autograd.grad(dF.sum(),data,create_graph=True)
        return solenoidal_field

    def smooth_relu(self, x):
        relu = x.relu()
        # TODO: Is there a clean way to avoid computing both of these on all elements?
        sq = (2*self._d*relu.pow(3) -relu.pow(4)) / (2 * self._d**3)
        lin = x - self._d/2
        return torch.where(relu < self._d, sq, lin)

# x = torch.Tensor(10, 2).uniform_(-10, 10)
# g = LyapunovFunction(2,6,2,(2,),0.1,[6,1],eps=0.0003)
# v,u = g(x)
# v = v.T
# ws = g._icnn._ws
# bs = g._icnn._bs
# us = g._icnn._us
# smooth = g.smooth_relu
# input_shape = (2,)
def lya(ws,bs,us,smooth,x,input_shape):
    if len(x.shape) < 2:
        x = x.unsqueeze(0)
    else:
        data_dims = list(range(1, len(input_shape) + 1))
        x = x.permute(*data_dims, 0)
    z = smooth(torch.addmm(bs[0],ws[0], x))
    for i in range(len(us)):
        u = F.softplus(us[i])
        w = ws[i + 1]
        b = bs[i + 1]
        z = smooth(torch.addmm(b, w, x) + torch.mm(u, z))
    return z

def gradient(outputs, inputs, grad_outputs=None, retain_graph=None, create_graph=False):
    r'''
    Compute the gradient of `outputs` with respect to `inputs`
    ```
    gradient(x.sum(), x)
    gradient((x * y).sum(), [x, y])
    ```
    '''
    if torch.is_tensor(inputs):
        inputs = [inputs]
    else:
        inputs = list(inputs)
    grads = torch.autograd.grad(outputs, inputs, grad_outputs,
                                allow_unused=True,
                                retain_graph=retain_graph,
                                create_graph=create_graph)
    grads = [x if x is not None else torch.zeros_like(y) for x, y in zip(grads, inputs)]
    return torch.cat([x.contiguous().view(-1) for x in grads])


def hessian(output, inputs, out=None, allow_unused=False, create_graph=False):
    r'''
    Compute the Hessian of `output` with respect to `inputs`
    ```
    hessian((x * y).sum(), [x, y])
    ```
    '''
    assert output.ndimension() == 0

    if torch.is_tensor(inputs):
        inputs = [inputs]
    else:
        inputs = list(inputs)

    numel = sum(p.numel() for p in inputs)
    if out is None:
        out = output.new_zeros(numel, numel)

    row_index = 0
    for i, inp in enumerate(inputs):
        [grad] = torch.autograd.grad(output, inp, create_graph=True, allow_unused=allow_unused)
        grad = torch.zeros_like(inp) if grad is None else grad
        grad = grad.contiguous().view(-1)

        for j in range(inp.numel()):
            if grad[j].requires_grad:
                row = gradient(grad[j], inputs[i:], retain_graph=True, create_graph=create_graph)[j:]
            else:
                row = grad[j].new_zeros(sum(x.numel() for x in inputs[i:]) - j)

            out[row_index, row_index:].add_(row.type_as(out))  # row_index's row
            if row_index + 1 < numel:
                out[row_index + 1:, row_index].add_(row[1:].type_as(out))  # row_index's column
            del row
            row_index += 1
        del grad

    return