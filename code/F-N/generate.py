import numpy as np
import math
import torch
import timeit 
# from harmonic_algo2 import *
from Control_Nonlinear_Icnn import *
import matplotlib.pyplot as plt
torch.set_default_dtype(torch.float64)

start = timeit.default_timer()
# np.random.seed(12)
#harmonic linear oscillator


# # dx_1 = x_1(1-x_1+2x_2)dt; dx_2 = x_2(1-2x_2+2x_1)dt
# x0 = [0.3,0.5]
# # x = torch.zeros([1,2])
# # print(x0[:,1])
# X = []
# X.append(x0)
# beta = 0.5


def FN_f(x):
    epi = 0.1
    beta = 0.7
    mu = 0.8
    I = 1.
    N = 100 # dimension
    y = torch.zeros_like(x)
    v = x[:,0:N:2]
    w = x[:,1:N:2]
    y[:,0:N:2] = v - v ** 3 / 3 - w + I
    y[:,1:N:2] = epi*(v+beta-mu*w)
    return y

def FN_g(x):
    k = 1./3.
    y = torch.zeros_like(x)
    syn_matrix = torch.from_numpy(np.load('./data/laplace_matrix.npy'))
    N = 100
    v = x[:,0:N:2]
    y[:,0:N:2] = k*torch.mm(syn_matrix,v.T).T
    return y

def FN_Jf(x):
    mu = 0.8
    epi = 0.1
    N = 100
    num = int(N/2)
    jacob = torch.kron(torch.eye(num),torch.tensor([[1.,-1.],[epi,-epi*mu]])).T
    ds = torch.mm(x,jacob)
    return ds

def FN_Jg(x):
    A = np.load('./data/laplace_matrix.npy')
    N = 100
    num = int(N/2)
    k = 1./3.
    jacob = torch.kron(torch.tensor(A),k*torch.tensor([[1.,0.],[0.,0.]])).T
    ds = torch.mm(x,jacob)
    return ds

def proj_es(x,u,f,g,model):
    D_in = 100
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


def run(N,dt,x0,seed,control=False):
    D_in = 100
    num = int(D_in/2)
    H1 = 200
    D_out = 100
    eps = 0.001
    model = LyapunovFunction(D_in,H1,D_out,(D_in,),0.1,[100,100,1],eps)
    model._icnn.load_state_dict(torch.load('./data/V_0.1.pkl'))
    model._control.load_state_dict(torch.load('./data/control_0.1.pkl'))
    model._mono.load_state_dict(torch.load('./data/class_k_0.1.pkl'))
    np.random.seed(seed)
    noise = np.random.normal(0, 1, N)
    X = torch.zeros([N,D_in])
    X[0] = torch.from_numpy(x0)
    for i in range(N-1):
        x = X[i]
        f = FN_f(x.view(-1, D_in))
        g = FN_g(x.view(-1, D_in))
        if control:
            t_x = x-x[0:2].repeat(50)
            if torch.max(torch.abs(t_x))<1e-5:
                u_es = torch.tensor(0.)
            else:
                with torch.no_grad():
                    u = model._control(t_x)*t_x.view(-1,D_in)
                    Jf = FN_Jf(t_x.view(-1,D_in)).detach()
                    Jg = FN_Jg(t_x.view(-1,D_in)).detach()
                u_sa = proj_sa(t_x.view(-1,D_in),u,Jf,Jg,model).detach()
                u_es = proj_es(t_x.view(-1, D_in), u_sa, Jf, Jg, model).detach()[0]
            new_x = x + dt * (f+u_es) + np.sqrt(dt) * g * noise[i]
        else:
            new_x = x + dt * (f) + np.sqrt(dt) * g * noise[i]
        X[i+1] = new_x
        if i%1000==0:
            print(i)
        # if torch.isnan(new_x[0,0]):
        #     print('nan number check: ',i,t_x[-1])
        #     break


    return X 

def generate(m,N,dt):
    seed_list = [1,4,5,9,15]
    D_in = 100
    num = int(D_in/2)
    W = torch.zeros(m, N, D_in)
    x0 = np.zeros([D_in])
    x0[0:D_in:2] = np.linspace(-2,2,num)
    x0[1:D_in:2] = np.linspace(0,2,num)
    for r in range(m):
        seed = seed_list[r]
        W[r,:] = run(N,dt,x0,seed,True)
        print(r)
    W = W.detach().numpy()
    np.save('./data/control_trajectory',W)
    # np.save('./data/uncontrol_trajectory', W)

T = 100
dt = 0.01
N = int(T/dt)
D_in = 100
num = int(D_in / 2)
x0 = np.zeros([D_in])
x0[0:D_in:2] = np.linspace(-2, 2, num)
x0[1:D_in:2] = np.linspace(0, 2, num)
# X = run(N,dt,x0,1).detach().numpy()  #seed: 1,4,5,9,15
generate(5,N,dt)
stop = timeit.default_timer()
print('total time:',stop-start)

# for i in range(50):
#     plt.plot(np.arange(N) * dt, X[:, 2 * i]-X[:,0])  # time trajectories
# plt.axhline(5,ls='--')
plt.show()







