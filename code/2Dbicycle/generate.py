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
    x,y,theta,v = state[:,0:1],state[:,1:2],state[:,2:3],state[:,3:4]
    dx = v*torch.cos(theta)
    dy = v*torch.sin(theta)
    dtheta = v
    dv = x**2+y**2
    ds = torch.cat((dx,dy,dtheta,dv),dim=1)
    return ds

def diff(state):
    k = 1.
    x,y,theta,v = state[:, 0], state[:, 1], state[:, 2], state[:, 3]
    ds = torch.zeros_like(state)
    ds[:,0]=k*x
    ds[:,1]=k*y
    return ds





def osqp(state,f,g,epi=0.5,p=10.0,gamma=2.):
    outer = 2.
    x,y,theta,v = state[0] , state[1], state[2], state[ 3]
    t_x = torch.from_numpy(state).clone().detach().requires_grad_(True)
    ZBF_h = outer**2-torch.sum(t_x[0:2]**2)
    hx = torch.autograd.grad(1 / ZBF_h, t_x, create_graph=True)[0]
    g_hxx = torch.autograd.grad(torch.sum(hx*torch.from_numpy(g)), t_x, create_graph=True)[0].detach().numpy()
    ZBF_h = ZBF_h.detach().numpy()
    hx = hx.detach().numpy()
    P = matrix(np.diag([2.0,2.0,2.0,2.0,2*p,2*p]))
    q = matrix([0.0,0.0,0.0,0.,0.,0.])
    G = matrix(np.array([[x,y,theta,v,-1.0,0.],[hx[0],hx[1],0.,0.,0.,-1.]]))
    h = matrix(np.array([[-state.dot(f)-0.5*g.dot(g)-epi/2*state.dot(state)],[gamma*(ZBF_h)-hx[0]*x-hx[1]*y-0.5*g.dot(g_hxx)]]))
    solvers.options['show_progress']=False
    sol=solvers.qp(P,q,G,h)
    u =np.array(sol['x'])
    return u[0:4,0]

def rl(state):
    #GP-MPC
    H = 5 # mpc iteration steps
    itr = 10 # gradient descent steps
    lr = 2. # learning rate
    D_in = 4
    p1 = 5.0 # weight factor for stability
    p2 = 2.5 # weight factor for safety
    p3 = 0.5
    outer = 2.
    A = torch.tensor([[0, 0, 0, 1.], [0,0,0, 0], [0, 0, 0, 1.],
                  [0,0,0,0]])
    def forward(state,u_seq):
        dt = 0.01
        mu,sigma = torch.zeros([H,D_in]),torch.zeros([H,D_in,D_in]) # moment forward
        dmu, dsigma = torch.zeros_like(mu), torch.zeros_like(sigma) # adjoint state
        mu[0] = state
        for i in range(H-1):
            x = mu[i].clone().detach().requires_grad_(True)
            mu[i+1] = mu[i]+dt*drift(mu[i:i+1])+dt*u_seq[i]
            sigma[i+1] = sigma[i]+dt*torch.eye(D_in)*torch.diag(diff(mu[i:i+1]))**2\
                         +dt**2*torch.mm(torch.mm(A,sigma[i]),A.T)
            h = F.relu(torch.sum(x[0:2])-outer**2)
            dmu[i] = p1*mu[i]+p2*torch.autograd.grad(h, x, create_graph=True)[0]
            dsigma[i]=p3*sigma[i]
        dmu[-1],dsigma[-1] = p1*mu[-1],p3*sigma[-1]
        dE = torch.zeros([H-1,D_in]) # derivative of Hamiltonian Energy
        for i in range(H-1):
            u = u_seq[i].clone().detach().requires_grad_(True)
            # l_MM = F.relu(torch.sin(mu[i][0])-0.5)+0.5*torch.sum(sigma[i]**2)  # running cost term
            co_energy = torch.dot(dmu[i+1],mu[i]+dt*drift(mu[i:i+1])[0]+dt*u)# +torch.sum(dsigma[i+1],sigma[i+1]) # multiplier term
            dE[i] = torch.autograd.grad(co_energy,u)[0]
        return dE
    u_ = torch.from_numpy(np.random.uniform(-1,1,[H-1,D_in])) # init
    for j in range(itr):
        u_ += -lr*forward(state,u_)
    return u_[0]

def proj_es(x,u,f,g,model):
    D_in = 4
    kappa = 0.5
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
    rho0 = np.random.uniform(0,2,1).item()
    angle0 = np.random.uniform(np.pi/6,12*np.pi/6,1).item()
    x0 = np.concatenate((np.array([rho0*np.cos(angle0),rho0*np.sin(angle0)]),np.random.uniform(-3,3,2)))
    X[0] = torch.from_numpy(x0)
    for i in range(N-1):
        x = X[i]
        f = drift(x.view(-1, D_in)).detach()
        g = diff(x.view(-1, D_in)).detach()
        t_x = x
        if mode=='fessc':
            if torch.max(torch.abs(t_x))<1e-5:
                u_es = torch.tensor(0.)
            else:
                with torch.no_grad():
                    u = model._control(t_x)*t_x.view(-1,D_in)
                    # Jf = t_drift(t_x.view(-1,D_in)).detach()
                    # Jg = t_diff(t_x.view(-1,D_in)).detach()
                u_sa = proj_sa(t_x.view(-1,D_in),u,f,g,model).detach()
                u_es = proj_es(t_x.view(-1, D_in), u_sa, f, g, model).detach()[0]
            new_x = x + dt * (f+u_es) + np.sqrt(dt) * g * noise[i]
            # new_x = x + dt * (f + u.detach().numpy()[0]) + np.sqrt(dt) * g * noise[i]
        elif mode == 'balsa':
            u = torch.from_numpy(osqp(t_x.detach().numpy(),f.numpy()[0],g.numpy()[0]))
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
    seed_list = [3,5,6,9,10]
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

T = 20
dt = 0.01
N = int(T/dt)



mode_list = ['orig','fessc','rl','balsa']
# X = run(N,dt,3,mode_list[2]).detach().numpy()  #seed: 3,5,6,9,10,11
generate(5,N,dt)
# stop = timeit.default_timer()
# print('total time:',stop-start)

# plt.plot(np.arange(N) * dt, np.sin(X[:, 0]))
# plt.plot(X[:,0],X[:,1])
# plt.plot(np.arange(N) * dt, np.sum(X[:, 0:2]**2,axis=1))
# plt.axhline(0)
# plt.axhline(-1)
# plt.axhline(4,ls='--')
# plt.axhline(np.pi,ls='--')
plt.show()







