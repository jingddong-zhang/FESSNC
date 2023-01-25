import torch 
import torch.nn.functional as F
import numpy as np
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

def FN_f(state,u):
    beta = 0.7
    mu = 0.8
    I = 1.
    epi = 0.1
    N = 100
    num = int(N/2)
    jacob = torch.kron(torch.eye(num),torch.tensor([[1.,-1.],[epi,-epi*mu]])).T
    ds = torch.mm(state,jacob)
    return ds+u

def FN_g(state,u):
    A = np.load('./data/laplace_matrix.npy')
    N = 100
    num = int(N/2)
    k = 1./3.
    jacob = torch.kron(torch.tensor(A),k*torch.tensor([[1.,0.],[0.,0.]])).T
    ds = torch.mm(state,jacob)
    return ds
    

'''
For learning 
'''
N = 500             # sample size
D_in = 100            # input dimension
H1 = 200             # hidden dimension
D_out = 100           # output dimension
x = torch.Tensor(N, D_in).uniform_(-5, 5)

# print(x.shape,FN_f(x,x).shape,FN_g(x,x).shape)

eps = 0.001 # L2 regularization coef
kappa = 0.1 # 指数稳定性系数
out_iters = 0
ReLU = torch.nn.ReLU()
# valid = False


while out_iters < 1: 
    # break
    start = timeit.default_timer()
    model = LyapunovFunction(D_in,H1,D_out,(D_in,),0.1,[100,100,1],eps)
    i = 0 
    t = 0 
    max_iters = 300
    learning_rate = 0.1
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    L = []

    r_f = torch.from_numpy(np.random.normal(0,1,D_in)) # hutch estimator vector
    while i < max_iters:
        # break
        # start = timeit.default_timer()
        output, u , alpha_h = model(x)

        f = FN_f(x,u)
        g = FN_g(x,u)


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

        stable_risk = (F.relu(L_V / num_V+kappa)).mean()
        safe_risk = (F.relu(-L_V-alpha_h)).mean()
        control_size = torch.sum(u**2)/N
        loss = stable_risk+control_size+safe_risk

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
        if loss < 1:
            break

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
