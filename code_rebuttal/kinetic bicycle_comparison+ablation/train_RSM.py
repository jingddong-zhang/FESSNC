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


'''
For learning 
'''

D_in = 4  # input dimension
H1 = 12  # hidden dimension
D_out = 4  # output dimension
n = 10
m = 5 # successive sample number
dt = 0.01

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

eps = 0.001  # L2 regularization coef
kappa = 0.1  # tolerance of rsm inequality
out_iters = 0
ReLU = torch.nn.ReLU()


mode = 2 # 1 for the original method in paper in https://ojs.aaai.org/index.php/AAAI/article/view/20695
         # 2 for the ICNN V function in our FESSNC paper
while out_iters < 1:
    # break
    start = timeit.default_timer()
    model = RSMFunction(D_in, H1, D_out,(D_in,), 0.1, [12, 12, 1], eps,mode)

    i = 0
    t = 0
    max_iters = 500
    learning_rate = 0.01
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    L = []
    tau = 6/n # mesh size
    if mode == 1:
        batch_size = 500
    elif mode == 2:
        batch_size = 100 # we set smaller batch size because the memory needed by ICNN is significantly larger than standard mlp
        #set 500 to estimate the training time used in table
    while i < max_iters:
        # break
        sub_start = timeit.default_timer()
        output, u = model(data)

        f = drift(data, u)
        g = diff(data, u)
        if mode == 1:
            lip_V = 2*model._V.calcul_lip()
        elif mode == 2:
            lip_V = model._V.calcul_lip() + eps*2*3
        lip_u = 1.0 # we use spectral norm regularization method to constrain the lipschitzian constant of the controller
        sub_lip = (1+1+6*dt+dt*lip_u)
        K = lip_V*sub_lip
        loss_rsm = 0.0
        s = torch.from_numpy(np.random.choice(np.arange(len(data), dtype=np.int64), batch_size, replace=False))
        for t in range(batch_size):
            j = s[t]
            x = data[j]
            pred_V = 0.
            for k in range(m):
                noise = torch.from_numpy(np.random.normal(0, 1, [1]))
                new_x = x + dt * f[j] + math.sqrt(dt)*noise*g[j]
                pred_V += model._V(new_x.view(-1,4))
            loss_rsm += F.relu(pred_V.mean()-output[j]+tau*K)
        loss_lip = F.relu(lip_V-kappa/(tau*sub_lip))
        loss = loss_rsm/len(data) + loss_lip

        L.append(loss)
        print(i, 'total loss=', loss.item(), "RSM score=", loss_rsm.item(),'Lipschitz constant:', loss_lip.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if loss_lip<1e-5 and loss<1e-5:
            break
        sub_stop = timeit.default_timer()
        print('per:',sub_stop-sub_start)
        print('Estimated total:', max_iters*(sub_stop - sub_start))
        i += 1

    # torch.save(model._control.state_dict(),'./data/control_rsm_{} mode={}.pkl'.format(kappa,mode))
    stop = timeit.default_timer()

    print('\n')
    print("Total time: ", stop - start)

    out_iters += 1
