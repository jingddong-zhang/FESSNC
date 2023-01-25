import numpy as np
import math
import torch
import timeit 
from cvxopt import solvers,matrix
from Control_Nonlinear_Icnn import *
import matplotlib.pyplot as plt
import torch.nn.functional as F
torch.set_default_dtype(torch.float64)

start = timeit.default_timer()
# np.random.seed(12)




def drift(state):
    g = 9.81
    L1,L2=1.,1.
    m1,m2=1.,1.
    theta1, z1, theta2, z2 = state[:,0:1],state[:,1:2],state[:,2:3],state[:,3:4]
    c, s = torch.cos(theta1 - theta2), torch.sin(theta1 - theta2)
    theta1dot = z1
    z1dot = (m2 * g * torch.sin(theta2) * c - m2 * s * (L1 * z1 ** 2 * c + L2 * z2 ** 2) -
             (m1 + m2) * g * torch.sin(theta1)) / L1 / (m1 + m2 * s ** 2)
    theta2dot = z2
    z2dot = ((m1 + m2) * (L1 * z1 ** 2 * s - g * torch.sin(theta2) + g * torch.sin(theta1) * c) +
             m2 * L2 * z2 ** 2 * s * c) / L2 / (m1 + m2 * s ** 2)
    ds = torch.cat((theta1dot, z1dot, theta2dot, z2dot),dim=1)
    return ds

def diff(state):
    k = 1.
    theta1, z1, theta2, z2 = state[:, 0], state[:, 1], state[:, 2], state[:, 3]
    ds = torch.zeros_like(state)
    ds[:,1]=torch.sin(theta1)*k
    ds[:,3]=torch.sin(theta2)*k
    return ds


def t_drift(state):
    g = 9.81
    L1,L2=1.,1.
    m1,m2=1.,1.
    theta1, z1, theta2, z2 = state[:,0:1]+math.pi,state[:,1:2],state[:,2:3]+math.pi,state[:,3:4]
    c, s = torch.cos(theta1 - theta2), torch.sin(theta1 - theta2)
    theta1dot = z1
    z1dot = (m2 * g * torch.sin(theta2) * c - m2 * s * (L1 * z1 ** 2 * c + L2 * z2 ** 2) -
             (m1 + m2) * g * torch.sin(theta1)) / L1 / (m1 + m2 * s ** 2)
    theta2dot = z2
    z2dot = ((m1 + m2) * (L1 * z1 ** 2 * s - g * torch.sin(theta2) + g * torch.sin(theta1) * c) +
             m2 * L2 * z2 ** 2 * s * c) / L2 / (m1 + m2 * s ** 2)
    ds = torch.cat((theta1dot, z1dot, theta2dot, z2dot),dim=1)
    return ds

def t_diff(state):
    k = 1.
    theta1, z1, theta2, z2 = state[:, 0]+math.pi, state[:, 1], state[:, 2]+math.pi, state[:, 3]
    ds = torch.zeros_like(state)
    ds[:,1]=torch.sin(theta1)*k
    ds[:,3]=torch.sin(theta2)*k
    return ds


def osqp(state,f,g,epi=0.5,p=10.0,gamma=2.):
    theta1, z1, theta2, z2 = state[0] , state[1], state[2], state[ 3]
    P = matrix(np.diag([2.0,2.0,2.0,2.0,2*p,2*p]))
    q = matrix([0.0,0.0,0.0,0.,0.,0.])
    G = matrix(np.array([[theta1,z1,theta2,z2,-1.0,0.],[np.cos(theta1)/(0.5-np.sin(theta1))**2,0.,0.,0.,0.,-1.]]))
    h = matrix(np.array([[-state.dot(f)-0.5*g.dot(g)-epi/2*state.dot(state)],[gamma*(0.5-np.sin(theta1))-np.cos(theta1)*theta1/(0.5-np.sin(theta1))**2]]))
    solvers.options['show_progress']=False
    sol=solvers.qp(P,q,G,h)
    u =np.array(sol['x'])
    return u[0:4,0]

def rl(state):
    #GP-MPC
    H = 10 # mpc iteration steps
    itr = 10 # gradient descent steps
    lr = 0.1 # learning rate
    D_in = 4
    g = 9.81
    p1 = 6.0 # weight factor for stability
    p2 = 2.5 # weight factor for safety
    p3 = 0.5
    L1, L2 = 1., 1.
    m1, m2 = 1., 1.
    A = torch.tensor([[0, 1.0, 0, 0], [(m1 + m2) * g / (m1 * L1), 0, -m2 * g / (m1 * L1), 0], [0, 0, 0, 1.],
                  [-g * (m1 + m2) / (m1 * L2), 0, g * (m1 + m2) / (m1 * L2), 0]])
    def forward(state,u_seq):
        dt = 0.01
        mu,sigma = torch.zeros([H,D_in]),torch.zeros([H,D_in,D_in]) # moment forward
        dmu, dsigma = torch.zeros_like(mu), torch.zeros_like(sigma) # adjoint state
        mu[0] = state
        for i in range(H-1):
            x = mu[i].clone().detach().requires_grad_(True)
            mu[i+1] = mu[i]+dt*t_drift(mu[i:i+1])+dt*u_seq[i]
            sigma[i+1] = sigma[i]+dt*torch.eye(D_in)*torch.diag(t_diff(mu[i:i+1]))**2\
                         +dt**2*torch.mm(torch.mm(A,sigma[i]),A.T)
            h = F.relu(torch.sin(x[0])-0.5)
            dmu[i] = p1*mu[i]+p2*torch.autograd.grad(h, x, create_graph=True)[0]
            dsigma[i]=p3*sigma[i]
        dmu[-1],dsigma[-1] = p1*mu[-1],p3*sigma[-1]
        dE = torch.zeros([H-1,D_in]) # derivative of Hamiltonian Energy
        for i in range(H-1):
            u = u_seq[i].clone().detach().requires_grad_(True)
            # l_MM = F.relu(torch.sin(mu[i][0])-0.5)+0.5*torch.sum(sigma[i]**2)  # running cost term
            co_energy = torch.dot(dmu[i+1],mu[i]+dt*t_drift(mu[i:i+1])[0]+u)# +torch.sum(dsigma[i+1],sigma[i+1]) # multiplier term
            dE[i] = torch.autograd.grad(co_energy,u)[0]
        return dE
    u_ = torch.from_numpy(np.random.uniform(-1,1,[H-1,D_in])) # init
    for j in range(itr):
        u_ += -lr*forward(state,u_)
    return u_[0]

def proj_es(x,u,f,g,model):
    D_in = 4
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
    return y

def run(N,dt,seed,mode='orig'):
    D_in = 4
    H1 = 12
    D_out = 4
    eps = 0.001
    model = LyapunovFunction(D_in,H1,D_out,(D_in,),0.1,[12,12,1],eps)
    model._icnn.load_state_dict(torch.load('./data/V_0.1.pkl'))
    model._control.load_state_dict(torch.load('./data/control_0.1.pkl'))
    model._mono.load_state_dict(torch.load('./data/class_k_0.1.pkl'))
    np.random.seed(seed)
    noise = np.random.normal(0, 1, [N,1])
    X = torch.zeros([N,D_in])
    x0 = np.random.uniform(-np.pi / 3, 4 * np.pi / 3, 4)  # initial value
    X[0] = torch.from_numpy(x0)
    for i in range(N-1):
        x = X[i]
        f = drift(x.view(-1, D_in)).detach()
        g = diff(x.view(-1, D_in)).detach()
        t_x = x - torch.tensor([math.pi, 0., math.pi, 0.])
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
        elif mode == 'balsa':
            Jf = t_drift(t_x.view(-1, D_in)).detach().numpy()[0]
            Jg = t_diff(t_x.view(-1, D_in)).detach().numpy()[0]
            u = torch.from_numpy(osqp(t_x.detach().numpy(),Jf,Jg))
            new_x = x + dt * (f+u) + np.sqrt(dt) * g * noise[i]
        elif mode=='rl':
            u = rl(t_x)
            new_x = x + dt * (f+u) + np.sqrt(dt) * g * noise[i]
        elif mode=='orig':
            new_x = x + dt * (f) + np.sqrt(dt) * g * noise[i]
        X[i+1] = new_x
        if i%500==0:
            print(i)


    return X 

def generate(m,N,dt):
    seed_list = [1,4,6,8,9]
    D_in = 4
    W = torch.zeros([4,m, N, D_in])
    time = torch.zeros([4,m])
    mode_list = ['orig', 'fessc', 'rl', 'balsa']
    for k in range(4):
        for r in range(m):
            seed = seed_list[r]
            start = timeit.default_timer()
            W[k,r,:] = run(N,dt,seed,mode_list[k])
            stop = timeit.default_timer()
            time[k,r]=stop-start
    W = W.detach().numpy()
    np.save('./data/trajectory_data',W)
    np.save('./data/time_cost',time)

T = 30
dt = 0.01
N = int(T/dt)

mode_list = ['orig','fessc','rl','balsa']
# X = run(N,dt,9,mode_list[3]).detach().numpy()  #seed: 1,4,6,8,9,14,15,16
generate(5,N,dt)
# stop = timeit.default_timer()
# print('total time:',stop-start)

# plt.plot(np.arange(N) * dt, np.sin(X[:, 0]))
# plt.plot(np.arange(N) * dt, X[:, 0])
# plt.axhline(0)
# plt.axhline(-1)
# plt.axhline(-np.pi/3,ls='--')
# plt.axhline(np.pi,ls='--')
plt.show()







