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
    x, y, theta, v = state[:, 0:1], state[:, 1:2], state[:, 2:3], state[:, 3:4]
    dx = v * torch.cos(theta)
    dy = v * torch.sin(theta)
    dtheta = v
    dv = x ** 2 + y ** 2
    ds = torch.cat((dx, dy, dtheta, dv), dim=1)
    return ds + u


def diff(state, u):
    k = 1.
    x, y, theta, v = state[:, 0], state[:, 1], state[:, 2], state[:, 3]
    ds = torch.zeros_like(state)
    ds[:, 0] = k * x
    ds[:, 1] = k * y
    return ds

def get_batch(data,batch_size):
    s=torch.from_numpy(np.random.choice(np.arange(len(data),dtype=np.int64),batch_size,replace=False))
    batch_x=data[s,:]  # (M, D)
    return batch_x

'''
For learning 
'''
N = 500  # sample size
D_in = 4  # input dimension
H1 = 12  # hidden dimension
D_out = 4  # output dimension
# rho = torch.Tensor(N, 1).uniform_(0, 3)
# angle = torch.Tensor(N, 1).uniform_(math.pi / 6, 12 * math.pi / 6)
# others = torch.Tensor(N, D_in - 2).uniform_(-3, 3)
# x = torch.cat((rho * torch.cos(angle), rho * torch.sin(angle), others), dim=1)

n = 10
x = torch.linspace(-2,2,n)
y = torch.linspace(-2,2,n)
theta = torch.linspace(-3,3,n)
v = torch.linspace(-3,3,n)
data = torch.zeros([n,n,n,n,4])
for i in range(n):
    for j in range(n):
        for k in range(n):
            for l in range(n):
                data[i, j, k, l, 0] = x[i]
                data[i, j, k, l, 1] = y[i]
                data[i, j, k, l, 2] = theta[i]
                data[i, j, k, l, 3] = v[i]
data = data.view(-1,4) # mesh grid data
x = data.requires_grad_(True)
eps = 0.001  # L2 regularization coef
kappa = 0.8  # 指数稳定性系数
out_iters = 0
batch_size = 500
# valid = False

while out_iters < 1:
    # break
    start = timeit.default_timer()
    model = LyapunovFunction(D_in, H1, D_out, (D_in,), 0.1, [12, 12, 1], eps)
    i = 0
    t = 0
    max_iters = 500
    learning_rate = 0.05
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    L = []
    tau = 6 / n  # mesh size
    r_f = torch.from_numpy(np.random.normal(0,1,D_in)) # hutch estimator vector

    while i < max_iters:
        # break
        sub_start = timeit.default_timer()
        batch_x = get_batch(x,batch_size)
        _, u, alpha_h = model(batch_x)

        f = drift(batch_x, u)
        g = diff(batch_x, u)

        loss_stable = (2 - kappa) * ((batch_x * g) ** 2) - batch_x ** 2 * (2 * batch_x * f + g ** 2)
        stable_risk = (F.relu(-loss_stable)).mean()


        def h_func(x):
            h = torch.sum(ZBF(x))
            return torch.sum(h)

        h = torch.sum(ZBF(batch_x))
        hx = torch.autograd.grad(h, batch_x, create_graph=True)[0]
        # r_hxx = torch.autograd.grad(torch.sum(hx * r_f, dim=1).sum(), batch_x, create_graph=True)[0]
        # L_h = torch.sum(hx * f, dim=1) + 0.5 * torch.sum((torch.sum(g * r_f, dim=1).view(-1, 1) * g) * r_hxx, dim=1)

        g_v = g.view(-1,1)
        hxx = torch.autograd.functional.hessian(h_func, batch_x).view([len(g_v), len(g_v)])
        L_h = torch.sum(hx * f, dim=1) + 0.5 * torch.sum((torch.mm(hxx,g_v).view(-1,4) * g), dim=1)

        M = torch.max(4*model._mono.integrand(ZBF(batch_x).unsqueeze(1)))
        # M = 1.0
        safe_risk = (F.relu(-L_h - alpha_h+4*tau*M)).mean()
        # control_size = torch.sum(u ** 2) / N
        loss = stable_risk + safe_risk #control_size + 1.2 * safe_risk

        L.append(loss)
        print(i, 'total loss=', loss.item(), "Lyapunov Risk=", stable_risk.item(), 'safe_risk=', safe_risk.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if loss < 1e-5:
            break

        sub_stop = timeit.default_timer()
        print('per:',sub_stop-sub_start)
        print('Estimated total:', max_iters*(sub_stop - sub_start))
        i += 1
    torch.save(model._control.state_dict(),'./data/control_sync theta={}.pkl'.format(kappa))

    stop = timeit.default_timer()

    print('\n')
    print("Total time: ", stop - start)

    out_iters += 1
