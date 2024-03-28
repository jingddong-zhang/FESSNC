import torch
import torch.nn.functional as F
import numpy as np
import math
import timeit
# from ICNN import *
from Control_Nonlinear_Icnn import *

torch.set_default_dtype(torch.float64)


def setup_seed(seed):
    torch.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    # torch.cuda.manual_seed(seed)
    np.random.seed(seed)


setup_seed(10)


def drift(state, u):
    g = 9.81
    ds = torch.zeros_like(state)
    for i in range(len(state)):
        theta1, z1, theta2, z2, theta3, z3 = state[i, 0] + math.pi, state[i, 1], state[i, 2] + math.pi, state[i, 3], \
                                             state[i, 4] + math.pi, state[i, 5]
        c, s = torch.cos(theta1 - theta2), torch.sin(theta1 - theta2)
        A = torch.tensor([[4.0, 2.0, 1.0], [2.0, 3.0, 1.0], [1.0, 1.0, 2.0]])
        B = g * torch.tensor([[3.0, 2.0, 1.0]])
        M = A * torch.tensor([[1.0, torch.cos(theta2 - theta1), torch.cos(theta3 - theta1)],
                              [torch.cos(theta2 - theta1), 1.0, torch.cos(theta3 - theta2)],
                              [torch.cos(theta3 - theta1), torch.cos(theta3 - theta2), 1.0]])
        N = -A * torch.tensor([[0.0, z2 * torch.sin(theta2 - theta1), z3 * torch.sin(theta3 - theta1)],
                               [z1 * torch.sin(theta1 - theta2), 0.0, z3 * torch.sin(theta3 - theta2)],
                               [z1 * torch.sin(theta1 - theta3), z2 * torch.sin(theta2 - theta3), 0.0]])
        Q = -B * torch.sin(torch.tensor([[theta1, theta2, theta3]]))
        Q = Q.T
        dz = torch.mm(torch.linalg.inv(M), -torch.mm(N, torch.tensor([[z1], [z2], [z3]])) - Q).T[0]
        ds[i, :] = torch.tensor([z1, dz[0], z2, dz[1], z3, dz[2]])

    return ds + u


def diff(state, u):
    k = 1.
    theta1, z1, theta2, z2, theta3, z3 = state[:, 0] + math.pi, state[:, 1], state[:, 2] + math.pi, state[:, 3], state[
                                                                                                                 :,
                                                                                                                 4] + math.pi, state[
                                                                                                                               :,
                                                                                                                               5]
    ds = torch.zeros_like(state)
    ds[:, 1] = torch.sin(theta1) * k
    ds[:, 3] = torch.sin(theta2) * k
    ds[:, 5] = torch.sin(theta3) * k
    return ds


'''
For learning 
'''
N = 500  # sample size
D_in = 6  # input dimension
H1 = 3 * D_in  # hidden dimension
D_out = 6  # output dimension
theta1 = torch.Tensor(N, 1).uniform_(-4 * math.pi / 3, math.pi / 3)
others = torch.Tensor(N, D_in - 1).uniform_(-5, 5)
x = torch.cat((theta1, others), dim=1)

eps = 0.001  # L2 regularization coef
kappa = 0.1  # 指数稳定性系数
out_iters = 0
ReLU = torch.nn.ReLU()
# valid = False


while out_iters < 1:
    # break
    start = timeit.default_timer()
    model = LyapunovFunction(D_in, H1, D_out, (D_in,), 0.1, [12, 12, 1], eps)
    i = 0
    t = 0
    max_iters = 300
    learning_rate = 0.1
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    L = []

    r_f = torch.from_numpy(np.random.normal(0, 1, D_in))  # hutch estimator vector
    while i < max_iters:
        # break
        # start = timeit.default_timer()
        output, u, alpha_h = model(x)

        f = drift(x, u)
        g = diff(x, u)

        x = x.clone().detach().requires_grad_(True)
        ws = model._icnn._ws
        bs = model._icnn._bs
        us = model._icnn._us
        smooth = model.smooth_relu
        input_shape = (D_in,)
        V1 = lya(ws, bs, us, smooth, x, input_shape)
        V0 = lya(ws, bs, us, smooth, torch.zeros_like(x), input_shape)
        num_V = smooth(V1 - V0) + eps * x.pow(2).sum(dim=1)
        V = torch.sum(smooth(V1 - V0) + eps * x.pow(2).sum(dim=1))
        Vx = torch.autograd.grad(V, x, create_graph=True)[0]
        r_Vxx = torch.autograd.grad(torch.sum(Vx * r_f, dim=1).sum(), x, create_graph=True)[0]
        L_V = torch.sum(Vx * f, dim=1) + 0.5 * torch.sum((torch.sum(g * r_f, dim=1).view(-1, 1) * g) * r_Vxx, dim=1)

        h = torch.sum(ZBF(x))
        hx = torch.autograd.grad(h, x, create_graph=True)[0]
        r_hxx = torch.autograd.grad(torch.sum(hx * r_f, dim=1).sum(), x, create_graph=True)[0]
        L_h = torch.sum(hx * f, dim=1) + 0.5 * torch.sum((torch.sum(g * r_f, dim=1).view(-1, 1) * g) * r_hxx, dim=1)

        stable_risk = (F.relu(L_V + kappa * num_V)).mean()
        safe_risk = (F.relu(-L_h - alpha_h)).mean()
        control_size = torch.sum(u ** 2) / N
        loss = stable_risk + control_size + safe_risk

        L.append(loss)
        print(i, 'total loss=', loss.item(), "Lyapunov Risk=", stable_risk.item(), 'safe_risk=', stable_risk.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # stop = timeit.default_timer()
        # print('per:',stop-start)
        i += 1
    # print(q)
    torch.save(model._icnn.state_dict(), './data/V_0.1_v1.pkl')
    torch.save(model._control.state_dict(), './data/control_0.1_v1.pkl')
    torch.save(model._mono.state_dict(), './data/class_K_0.1_v1.pkl')
    stop = timeit.default_timer()

    print('\n')
    print("Total time: ", stop - start)

    out_iters += 1
