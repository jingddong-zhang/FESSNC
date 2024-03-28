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



kappa = 0.5  # 指数稳定性系数
out_iters = 0


def train():
    setup_seed(10)
    N = 500  # sample size
    D_in = 4  # input dimension
    H1 = 12  # hidden dimension
    D_out = 4  # output dimension
    rho = torch.Tensor(N, 1).uniform_(0, 3)
    angle = torch.Tensor(N, 1).uniform_(math.pi / 6, 12 * math.pi / 6)
    others = torch.Tensor(N, D_in - 2).uniform_(-3, 3)
    x = torch.cat((rho * torch.cos(angle), rho * torch.sin(angle), others), dim=1)
    eps = 0.001  # L2 regularization coef
    start = timeit.default_timer()
    model = LyapunovFunction(D_in, H1, D_out, (D_in,), 0.1, [12, 12, 1], eps)

    t = 0
    max_iters = 500
    learning_rate = 0.05
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    L = []
    r_f = torch.from_numpy(np.random.normal(0, 1, D_in))  # hutch estimator vector

    hyper_list = [0.1,0.5,1.0]
    for kappa in hyper_list:
        for lambda_1 in hyper_list:
            for lambda_2 in hyper_list:
                i = 0
                while i < max_iters:
                    # break
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

                    stable_risk = (F.relu(L_V / num_V + kappa)).mean()
                    safe_risk = (F.relu(-L_h - alpha_h)).mean()
                    control_size = torch.sum(u ** 2) / N
                    loss = control_size + lambda_1 * stable_risk + lambda_2 * safe_risk

                    L.append(loss)
                    if i%100 == 0:
                        print(i, 'kappa={} l1={} l2={}'.format(kappa,lambda_1,lambda_2),'total loss=', loss.item())
                        # print(i, 'total loss=', loss.item(), "Lyapunov Risk=", stable_risk.item(), 'safe_risk=', safe_risk.item(),
                        #         control_size.item())

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    i += 1

                torch.save(model._icnn.state_dict(),'./data/ablation_data/V k={} l1={} l2={}.pkl'.format(kappa,lambda_1,lambda_2))
                torch.save(model._control.state_dict(),'./data/ablation_data/control k={} l1={} l2={}.pkl'.format(kappa,lambda_1,lambda_2))
                torch.save(model._mono.state_dict(), './data/ablation_data/class_K k={} l1={} l2={}.pkl'.format(kappa,lambda_1,lambda_2))
                stop = timeit.default_timer()

                print('\n')
                print("Total time: ", stop - start)

# train()


def run_ablation(N,dt,seed,model):
    def drift(state):
        x, y, theta, v = state[:, 0:1], state[:, 1:2], state[:, 2:3], state[:, 3:4]
        dx = v * torch.cos(theta)
        dy = v * torch.sin(theta)
        dtheta = v
        dv = x ** 2 + y ** 2
        ds = torch.cat((dx, dy, dtheta, dv), dim=1)
        return ds

    def diff(state):
        k = 1.
        x, y, theta, v = state[:, 0], state[:, 1], state[:, 2], state[:, 3]
        ds = torch.zeros_like(state)
        ds[:, 0] = k * x
        ds[:, 1] = k * y
        return ds
    D_in = 4
    H1 = 12
    D_out = 4
    np.random.seed(seed)
    noise = np.random.normal(0, 1, [N,1])
    X = torch.zeros([N,D_in])
    U = torch.zeros([N,D_in])
    rho0 = np.random.uniform(0.9,2,1).item()
    angle0 = np.random.uniform(np.pi/6,12*np.pi/6,1).item()
    x0 = np.concatenate((np.array([rho0*np.cos(angle0),rho0*np.sin(angle0)]),np.random.uniform(-3,3,2)))
    X[0] = torch.from_numpy(x0)
    for i in range(N-1):
        x = X[i]
        f = drift(x.view(-1, D_in)).detach()
        g = diff(x.view(-1, D_in)).detach()
        t_x = x
        if torch.max(torch.abs(t_x)) < 1e-5:
            u_es = torch.tensor(0.)
        else:
            with torch.no_grad():
                u = model._control(t_x) * t_x.view(-1, D_in)
            u_sa = proj_sa(t_x.view(-1, D_in), u, f, g, model).detach()
            u_es = proj_es(t_x.view(-1, D_in), u_sa, f, g, model).detach()[0]
        new_x = x + dt * (f + u_es) + np.sqrt(dt) * g * noise[i]
        U[i + 1] = u_es


        X[i + 1] = new_x

        # if i%500==0:
        #     print(i)
    return X,U

def generate_ablation(N, dt):
    start = timeit.default_timer()
    D_in = 4
    H1 = 12
    D_out = 4
    eps = 0.001
    seed_list = [3, 6, 9, 10, 11, 12, 14, 15, 16, 28]  # initial range (0.9,2)
    m = len(seed_list)
    model = LyapunovFunction(D_in, H1, D_out, (D_in,), 0.1, [12, 12, 1], eps)
    hyper_list = [0.1, 0.5, 1.0]
    num_modes = len(hyper_list)
    W = torch.zeros([num_modes,num_modes,num_modes, m, N, D_in])
    U_ = torch.zeros([num_modes,num_modes,num_modes, m, N, D_in])
    for i in range(num_modes):
        for j in range(num_modes):
            for k in range(num_modes):
                    kappa,l1,l2 = hyper_list[i],hyper_list[j],hyper_list[k]
                    model._icnn.load_state_dict(torch.load('./data/ablation_data/V k={} l1={} l2={}.pkl'.format(kappa,l1,l2)))
                    model._control.load_state_dict(torch.load('./data/ablation_data/control k={} l1={} l2={}.pkl'.format(kappa,l1,l2)))
                    model._mono.load_state_dict(torch.load('./data/ablation_data/class_K k={} l1={} l2={}.pkl'.format(kappa,l1,l2)))
                    for r in range(m):
                        seed = seed_list[r]
                        X, U = run_ablation(N, dt, seed, model)
                        W[i,j,k, r, :] = X
                        U_[i,j,k, r, :] = U
                        print(f'current stat: {kappa,l1,l2,r} time={timeit.default_timer()-start}')
    W = W.detach().numpy()
    U_ = U_.detach().numpy()

    np.save('./data/ablation_data/trajectory_data_ablation', {'state': W, 'control': U_})

def analyze(state):
    v = np.linalg.norm(state[:,:,:,:,:,0:2],ord=2,axis=5)
    print(v.shape)
    hyper_list = [0.1, 0.5, 1.0]
    num_modes = len(hyper_list)
    v_min = np.min(v[:,:,:,:,:],axis=4)
    v_std = np.std(v[:,:,:,:,-500:],axis=4)
    v_safe = np.max(v[:,:,:,:,:200],axis=4)

    num = 0
    for k in range(num_modes):
        for j in range(num_modes):
            for i in range(num_modes):
                sub_min = v_min[i,j,k,:]
                sub_min = sub_min[np.where(np.isnan(sub_min)==False)[0]]
                sub_std = v_std[i,j,k,:]
                sub_std = sub_std[np.where(np.isnan(sub_std) == False)[0]]
                sub_safe = v_safe[i,j,k,:]
                sub_safe = sub_safe[np.where(np.isnan(sub_safe) == False)[0]]
                kappa, l1, l2 = hyper_list[i], hyper_list[j], hyper_list[k]
                # print(f'${sub_min.mean():.5f}$|',end='')
                # print(f'${sub_std.mean():.5f}$|', end='')
                print(f'${sub_safe.mean():.2f}$|', end='')
                num += 1
                if num % 9 == 0:
                    print('\n')


    # def success_rate(data):
    #     v = np.sqrt(data[:, :, :, 0] ** 2 + data[:, :, :, 1] ** 2)  # delta_theta
    #     v_slice = np.zeros([len(v), v.shape[1], L])
    #     v_num = np.zeros([len(v), L])
    #     for k in range(len(v)):
    #         for i in range(L):
    #             value = np.max(v[k, :, n * i:n * (i + 1)], axis=1)
    #             v_slice[k, :, i] = value
    #             index1 = np.where(value < 0.1)[0]
    #             index2 = np.where(s_tmax[k, :] < 2)[0]
    #             index = [_ for _ in index1 if _ in index2]
    #             v_num[k, i] = len(index)
    #     return v_num / v.shape[1] * 100
    #
    # v = success_rate(data)

T = 20
dt = 0.01
N = int(T/dt)

# generate_ablation(N,dt)
data = np.load('./data/ablation_data/trajectory_data_ablation.npy',allow_pickle=True).item()
W = data['state']
analyze(W)