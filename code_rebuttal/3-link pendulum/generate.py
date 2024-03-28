import numpy as np
import math
import torch
import timeit 
from cvxopt import solvers,matrix
from Control_Nonlinear_Icnn import *
from kernel_machine import *
import matplotlib.pyplot as plt
import torch.nn.functional as F
from scipy import integrate
torch.set_default_dtype(torch.float64)

start = timeit.default_timer()
# np.random.seed(12)
def drift(state):
    g = 9.81
    ds = torch.zeros_like(state)
    for i in range(len(state)):
        theta1, z1, theta2, z2, theta3, z3 = state[i, 0], state[i, 1], state[i, 2], state[i, 3], \
                                             state[i, 4], state[i, 5]
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
    return ds

def diff(state):
    k = 1.
    theta1, z1, theta2, z2, theta3, z3 = state[:, 0], state[:, 1], state[:, 2],\
                                         state[:, 3], state[:,4], state[:,5]
    ds = torch.zeros_like(state)
    ds[:, 1] = torch.sin(theta1) * k
    ds[:, 3] = torch.sin(theta2) * k
    ds[:, 5] = torch.sin(theta3) * k
    return ds

def t_drift(state):
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
    return ds

def t_diff(state):
    k = 1.
    theta1, z1, theta2, z2, theta3, z3 = state[:, 0] + math.pi, state[:, 1], state[:, 2] + math.pi,\
                                         state[:, 3], state[:,4] + math.pi, state[:,5]
    ds = torch.zeros_like(state)
    ds[:, 1] = torch.sin(theta1) * k
    ds[:, 3] = torch.sin(theta2) * k
    ds[:, 5] = torch.sin(theta3) * k
    return ds



def proj_es(x,u,f,g,model):
    D_in = 6
    kappa = 0.1
    x = x.clone().detach().requires_grad_(True)
    ws = model._icnn._ws
    bs = model._icnn._bs
    us = model._icnn._us
    smooth = model.smooth_relu
    input_shape = (D_in,)
    V1 = lya(ws, bs, us, smooth, x, input_shape)
    V0 = lya(ws, bs, us, smooth, torch.zeros_like(x), input_shape)
    V = smooth(V1 - V0) + model._eps * x.pow(2).sum(dim=1)
    # V = torch.sum(smooth(V1 - V0) + model._eps * x.pow(2).sum(dim=1))
    Vx = torch.autograd.grad(V, x, create_graph=True)[0]
    r_Vxx = torch.autograd.grad(torch.sum(Vx * g, dim=1).sum(), x, create_graph=True)[0]
    proj_u = u - Vx*torch.relu(torch.dot(Vx[0],(f+u)[0])+0.5*torch.dot(r_Vxx[0],g[0])+kappa*V)/torch.dot(Vx[0],Vx[0])
    return proj_u

def proj_sa(x,u,f,g,model):
    x = x.clone().detach().requires_grad_(True)
    h = ZBF(x)
    hx = torch.autograd.grad(h, x, create_graph=True)[0]
    r_hxx = torch.autograd.grad(torch.sum(hx * g, dim=1).sum(), x, create_graph=True)[0]
    proj_u = u + hx*torch.relu(-torch.dot(hx[0],(f+u)[0])-0.5*torch.dot(r_hxx[0],g[0])-model._mono(h.unsqueeze(1))[:, 0])/torch.dot(hx[0],hx[0])
    return proj_u

def commute(x):
    y = torch.zeros_like(x)
    y[:,1] = x[:,0]
    y[:,3] = x[:,2]
    y[:,5] = x[:,4]
    return y

def run(N,dt,seed,mode='orig'):
    D_in = 6  # input dimension
    H1 = 3 * D_in  # hidden dimension
    D_out = 6  # output dimension
    eps = 0.001
    model = LyapunovFunction(D_in,H1,D_out,(D_in,),0.1,[12,12,1],eps)
    model._icnn.load_state_dict(torch.load('./data/V_0.1_v1.pkl'))
    model._control.load_state_dict(torch.load('./data/control_0.1_v1.pkl'))
    model._mono.load_state_dict(torch.load('./data/class_k_0.1_v1.pkl'))
    modelkernel = rectified_flow
    mask = torch.tensor([[0.0,1.0,0.0,1.0,0.0,1.0]])
    np.random.seed(seed)
    noise = np.random.normal(0, 1, [N,1])
    X = torch.zeros([N,D_in])
    U = torch.zeros([N,D_in])
    x0 = np.random.uniform(-np.pi / 3, 4 * np.pi / 3, 6)  # initial value
    X[0] = torch.from_numpy(x0)
    for i in range(N-1):
        x = X[i]
        f = drift(x.view(-1, D_in)).detach()
        g = diff(x.view(-1, D_in)).detach()
        t_x = x - torch.tensor([math.pi, 0., math.pi, 0., math.pi, 0.])
        if mode=='fessc':
            if torch.max(torch.abs(t_x))<1e-5:
                u_es = torch.tensor(0.)
            else:
                with torch.no_grad():
                    u = model._control(t_x)*commute(t_x.view(-1,D_in))
                    Jf = t_drift(t_x.view(-1,D_in)).detach()
                    Jg = t_diff(t_x.view(-1,D_in)).detach()
                u_sa = proj_sa(t_x.view(-1,D_in),u,Jf,Jg,model).detach()
                u_es = proj_es(t_x.view(-1, D_in), u_sa, Jf, Jg, model).detach()[0]
            new_x = x + dt * (f+u_es) + np.sqrt(dt) * g * noise[i]
            U[i+1] = u_es
        elif mode=='kernel':
            Jf = t_drift(t_x.view(-1, D_in)).detach()
            u = modelkernel.controller(t_x,i/N,Jf)*mask
            u = u[0]
            new_x = x + dt * (f+u) + np.sqrt(dt) * g * noise[i]
            U[i + 1] = u
        elif mode=='fesskernel':
            Jf = t_drift(t_x.view(-1, D_in)).detach()
            Jg = t_diff(t_x.view(-1, D_in)).detach()
            u = modelkernel.controller(t_x,i/N,Jf)*mask
            u_sa = proj_sa(t_x.view(-1, D_in), u, Jf, Jg, model).detach()
            u_es = proj_es(t_x.view(-1, D_in), u_sa, Jf, Jg, model).detach()[0]
            new_x = x + dt * (f+u_es) + np.sqrt(dt) * g * noise[i]
            U[i + 1] = u_es
        elif mode=='orig':
            new_x = x + dt * (f) + np.sqrt(dt) * g * noise[i]
        X[i+1] = new_x
        if i%500==0:
            print(i)


    return X,U

def generate(N,dt):
    seed_list = [1,4,6,8,9,10,11,12,13,14]#[1,4,6,8,9]
    m = len(seed_list)
    D_in = 6
    mode_list = ['fessc','kernel','fesskernel']
    num_modes = len(mode_list)
    W = torch.zeros([num_modes,m, N, D_in])
    U_ = torch.zeros([num_modes, m, N, D_in])


    for k in range(num_modes):
        for r in range(m):
            seed = seed_list[r]
            X,U = run(N,dt,seed,mode_list[k])
            W[k,r,:] = X
            U_[k, r, :] = U

    W = W.detach().numpy()
    U_ = U_.detach().numpy()
    np.save('./data/trajectory_data_v1',{'state':W,'control':U_})



def analyaze(data,U):
    n = 200  # per trial : n time steps
    L = int(3000 / n)
    v = 2 * np.pi - (data[:, :, :, 0] + data[:, :, :, 2]) # the angle between the end tip to the vertical line, 0 angle is the down still state
    v = np.abs(v)
    v_max = np.max(data[:,:,:,0], axis=2)
    v_min = np.min(data[:,:,:,0], axis=2)
    def success_rate(data):
        v_slice = np.zeros([len(v), v.shape[1], L])
        v_num = np.zeros([len(v), L])
        for k in range(len(v)):
            for i in range(L):
                value = np.max(v[k, :, n * i:n * (i + 1)], axis=1)
                v_slice[k, :, i] = value
                index1 = np.where(value < np.pi / 40)[0] # success rate
                # index1 = [i for i in range(data.shape[1])] # safety rate
                index2 = np.where(v_max[k,:] <= math.pi*7/6)[0]
                index3 = np.where(v_min[k, :] >= -math.pi/6)[0]
                index = [_ for _ in index1 if _ in index2 and _ in index3]
                v_num[k, i] = len(index)
        return v_num / v.shape[1] * 100

    res = success_rate(data)
    print(res,'\n')


    std_stat = np.std(v[:,:,-200:],axis=2)
    std_list = []
    for i in range(len(std_stat)):
        sub_stat = std_stat[i]
        sub_stat = sub_stat[np.where(np.isnan(sub_stat)==False)[0]]
        std_list.append(np.mean(sub_stat))

    print(f'Variance statistics: {std_list}','\n')

    U = np.linalg.norm(U,ord=2,axis=3)
    t = np.linspace(0,30,U.shape[-1])
    energy_list = []
    for i in range(U.shape[0]):
        sub_energy = np.zeros([U.shape[1]])
        for j in range(U.shape[1]):
            if np.isnan(data[i,j,:,:].any())==False:
                energy = integrate.trapz(U[i,j,:],t[:])
            sub_energy[j] = energy
        energy_list.append(sub_energy.mean())
    print(f'energy statistics: {energy_list}')




T = 30
dt = 0.01
N = int(T/dt)
generate(N,dt)

data = np.load('./data/trajectory_data_v1.npy',allow_pickle=True).item()
X,U = data['state'],data['control']
analyaze(X,U)





