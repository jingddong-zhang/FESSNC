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

def drift(state,u):
    x,y,theta,v = state[:,0:1],state[:,1:2],state[:,2:3],state[:,3:4]
    dx = v*torch.cos(theta)
    dy = v*torch.sin(theta)
    dtheta = v
    dv = x**2+y**2
    ds = torch.cat((dx,dy,dtheta,dv),dim=1)
    return ds+u

def diff(state,u):
    k = 1.
    x,y,theta,v = state[:, 0], state[:, 1], state[:, 2], state[:, 3]
    ds = torch.zeros_like(state)
    ds[:,0]=k*x
    ds[:,1]=k*y
    return ds


'''
For learning 
'''
N = 500             # sample size
D_in = 4           # input dimension
H1 = 12             # hidden dimension
D_out = 4          # output dimension
rho = torch.Tensor(N, 1).uniform_(0,3)
angle = torch.Tensor(N, 1).uniform_(math.pi/6,12*math.pi/6)
others = torch.Tensor(N, D_in-2).uniform_(-3, 3)
x = torch.cat((rho*torch.cos(angle),rho*torch.sin(angle),others),dim=1)

eps = 0.001 # L2 regularization coef
kappa = 0.5 # 指数稳定性系数
out_iters = 0
ReLU = torch.nn.ReLU()
# valid = False


while out_iters < 1: 
    # break
    start = timeit.default_timer()
    model = LyapunovFunction(D_in,H1,D_out,(D_in,),0.1,[12,12,1],eps)
    i = 0 
    t = 0 
    max_iters = 500
    learning_rate = 0.05
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    L = []

    r_f = torch.from_numpy(np.random.normal(0,1,D_in)) # hutch estimator vector
    while i < max_iters:
        # break
        # start = timeit.default_timer()
        output, u , alpha_h = model(x)

        f = drift(x,u)
        g = diff(x,u)


        x = x.clone().detach().requires_grad_(True)
        ws = model._icnn._ws
        bs = model._icnn._bs
        us = model._icnn._us
        smooth = model.smooth_relu
        input_shape = (D_in,)
        V1 = lya(ws,bs,us,smooth,x,input_shape)
        V0 = lya(ws,bs,us,smooth,torch.zeros_like(x),input_shape)
        num_V = smooth(V1-V0)+eps*x.pow(2).sum(dim=1)
        V = torch.sum(smooth(V1-V0)+eps*x.pow(2).sum(dim=1))
        Vx = torch.autograd.grad(V, x, create_graph=True)[0]
        r_Vxx = torch.autograd.grad(torch.sum(Vx*r_f,dim=1).sum(), x, create_graph=True)[0]
        L_V = torch.sum(Vx*f,dim=1)+0.5*torch.sum((torch.sum(g*r_f,dim=1).view(-1,1)*g)*r_Vxx,dim=1)

        h = torch.sum(ZBF(x))
        hx = torch.autograd.grad(h, x, create_graph=True)[0]
        r_hxx = torch.autograd.grad(torch.sum(hx*r_f,dim=1).sum(), x, create_graph=True)[0]
        L_h = torch.sum(hx*f,dim=1)+0.5*torch.sum((torch.sum(g*r_f,dim=1).view(-1,1)*g)*r_hxx,dim=1)

        stable_risk = (F.relu(L_V/num_V+kappa)).mean()
        safe_risk = (F.relu(-L_V-alpha_h)).mean()
        control_size = torch.sum(u**2)/N
        loss = stable_risk+control_size+1.2*safe_risk

        L.append(loss)
        print(i, 'total loss=',loss.item(),"Lyapunov Risk=",stable_risk.item(),'safe_risk=',stable_risk.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step() 

        # q = model.control.weight.data.numpy()
        # if loss < 10.0:
        #     optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        # else:
        #     optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
        # if loss < 0.5:
        #     break

        # stop = timeit.default_timer()
        # print('per:',stop-start)
        i += 1
    # print(q)
    torch.save(model._icnn.state_dict(),'./data/V_0.1.pkl')
    torch.save(model._control.state_dict(),'./data/control_0.1.pkl')
    torch.save(model._mono.state_dict(), './data/class_K_0.1.pkl')
    stop = timeit.default_timer()


    print('\n')
    print("Total time: ", stop - start)
    
    out_iters+=1
