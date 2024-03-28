import matplotlib.pyplot as plt
import numpy as np
import torch

from Control_Nonlinear_Icnn import *

def rbf_kernel(a, b, h):
    norm = torch.norm(a-b, dim=-1, keepdim=False)
    log_kappa = (-1.) * norm.pow(2) / (2. * h)

    return log_kappa

class NW(nn.Module):
    def __init__(self, pairs, sample_size, bandwidth=1.):
        super().__init__()
        self.pairs = pairs
        self.number_of_pairs = self.pairs.shape[0]
        self.sample_size = sample_size
        self.h = bandwidth

    def forward(self, x_input, t):
        indices = torch.randperm(self.number_of_pairs)[:self.sample_size]
        pi_0_sample = self.pairs[indices, 0].detach().clone()
        pi_1_sample = self.pairs[indices, 1].detach().clone()
        pi_0_sample = pi_0_sample[None, :, :].repeat(x_input.shape[0], 1, 1)
        pi_1_sample = pi_1_sample[None, :, :].repeat(x_input.shape[0], 1, 1)
        x_input = x_input[:, None, :].repeat(1, self.sample_size, 1)
        x_t = t * pi_1_sample + (1.-t) * pi_0_sample
        log_kappa = rbf_kernel(x_t.detach().clone(), x_input.detach().clone(), self.h)
        omega = torch.softmax(log_kappa, dim=-1) * self.sample_size  ### multiplying self.sample_size because the denominator in softmax is sum, not mean

        v = (pi_1_sample - x_input) / (1.-t) * omega[:,:,None]
        v = v.mean(dim=1, keepdim=False)

        return v


class RectifiedFlow():
    def __init__(self, model=None, num_steps=1000):
        self.model = model
        self.N = num_steps

    @torch.no_grad()
    def sample_ode(self, z0=None, N=None):
        ### NOTE: Use Euler method to sample from the learned flow
        if N is None:
            N = self.N
        dt = 1. / N
        traj = []  # to store the trajectory
        z = z0.detach().clone()
        batchsize = z.shape[0]

        traj.append(z.detach().clone())
        for i in range(N):
            t = i / N
            pred = self.model(z, t)
            z = z.detach().clone() + pred * dt

            traj.append(z.detach().clone())

        return traj

    def controller(self, state=None, t=None,f=None):
        '''
        :param z: current state, size (1,D_in)
        :param t: t is the relative time, i.e., t=t_s/T, where t_s is the current real time and T is the total time
        :param f: the current drift term, size (1,D_in)
        :return:
        '''
        state = state.view(1,6)
        pred = self.model(state, t)
        # pred = pred # size (D_in,)
        control = pred - f
        return control

data_size = 10000
D_in = 6  # input dimension
theta1 = torch.Tensor(data_size, 1).uniform_(-4 * math.pi / 3, math.pi / 3)
others = torch.Tensor(data_size, D_in - 1).uniform_(-5, 5)
x_0 = torch.cat((theta1, others), dim=1)
x_1 = torch.zeros_like(x_0)
x_pairs = torch.stack([x_0, x_1], dim=1)
rectified_flow = RectifiedFlow(model=NW(pairs=x_pairs, sample_size=4096, bandwidth=0.001))
