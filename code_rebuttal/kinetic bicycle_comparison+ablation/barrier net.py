import timeit
from tqdm import tqdm
import numpy as np
import torch
from cvxopt import solvers,matrix
from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm
from Control_Nonlinear_Icnn import *
# Define computation as a nn.Module.

class NormalControlNet(torch.nn.Module):

    def __init__(self, n_input, n_hidden, n_output):
        super(NormalControlNet, self).__init__()
        torch.manual_seed(2)
        self.layer1 = (torch.nn.Linear(n_input, n_hidden))
        self.layer2 = (torch.nn.Linear(n_hidden, n_hidden))
        self.layer3 = (torch.nn.Linear(n_hidden, n_output))

    def forward(self, x):
        sigmoid = torch.nn.ReLU()
        h_1 = sigmoid(self.layer1(x))
        h_2 = sigmoid(self.layer2(h_1))
        out = self.layer3(h_2)
        return out

class MyModel(nn.Module):
    def __init__(self, n_input, n_hidden, n_output):
        super(MyModel, self).__init__()
        torch.manual_seed(2)
        self.net = NormalControlNet(n_input,n_hidden,n_output)
    def forward(self, x):
        # Define your computation here.
        return self.net(x)

def check():
    '''
    run this function to check if your device can carry out the auto-LiRPA package https://github.com/Verified-Intelligence/auto_LiRPA
    '''
    model = MyModel(2,3,2)
    # my_input = load_a_batch_of_data()
    my_input = torch.tensor([[2.0,2.0]])
    # Wrap the model with auto_LiRPA.
    model = BoundedModule(model, my_input)
    # Define perturbation. Here we add Linf perturbation to input data.
    ptb = PerturbationLpNorm(norm=np.inf, eps=0.1)
    # Make the input a BoundedTensor with the pre-defined perturbation.
    my_input = BoundedTensor(my_input, ptb)
    # Regular forward propagation using BoundedTensor works as usual.
    prediction = model(my_input)
    # Compute LiRPA bounds using the backward mode bound propagation (CROWN).
    lb, ub = model.compute_bounds(x=(my_input,), method="backward")
    print(lb,ub)

def lq(lb_tensor,ub_tensor):
    lb_tensor,ub_tensor = lb_tensor.detach().numpy(),ub_tensor.detach().numpy()
    u = np.zeros_like(lb_tensor)
    n = len(u)
    C = matrix([1., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0.])
    A1 = np.concatenate((-np.eye(4), np.eye(4), np.zeros([4, 4])), axis=1)
    A2 = np.concatenate((-np.eye(4), -np.eye(4), np.zeros([4, 4])), axis=1)
    A3 = np.concatenate((np.zeros([4, 4]), -np.eye(4), np.eye(4)), axis=1)
    A4 = np.concatenate((np.zeros([4, 4]), np.eye(4), -np.eye(4)), axis=1)
    A = np.concatenate((A1, A2, A3, A4), axis=0)
    A = matrix(A)
    B_ast = np.zeros([8])
    for i in tqdm(range(n)):
        for j in range(n):
            for k in range(n):
                for l in range(n):
                    lb,ub = lb_tensor[i,j,k,l],ub_tensor[i,j,k,l]
                    B = matrix(np.concatenate((B_ast,lb,ub))) # minimizer state=(0,0,0,0)
                    solvers.options['show_progress']=False
                    sol = solvers.lp(C, A, B)
                    if not sol['x']==None:
                        # print(sol['x'])
                        u[i, j, k, l, :] = np.array(sol['x'])[8:12, 0]
                    # u[i,j,k,l,:] = np.array(sol['x'])[8:12,0]
    return u
#
# u = np.zeros([7,7,7,7,4])
# C = matrix([1., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0.])
# A1 = np.concatenate((-np.eye(4), np.eye(4), np.zeros([4, 4])), axis=1)
# A2 = np.concatenate((-np.eye(4), -np.eye(4), np.zeros([4, 4])), axis=1)
# A3 = np.concatenate((np.zeros([4, 4]), -np.eye(4), np.eye(4)), axis=1)
# A4 = np.concatenate((np.zeros([4, 4]), np.eye(4), -np.eye(4)), axis=1)
# A = np.concatenate((A1, A2, A3, A4), axis=0)
# A = matrix(A)
# B = matrix(np.concatenate((np.zeros(8), np.zeros(8))))
# print(B)
# # A = matrix([[1.],[1.],[1.],[1.],[1.],[1.],[1.],[1.],[1.],[1.],[1.],[1.]])
# # B = matrix([1.0])
# # A = matrix([
# #     [-1., -1., 0., 1.],
# #     [1., -1., -1., -2.]
# # ])
# # # print(A.T)
# # A = matrix((np.random.uniform(0,1,[4,2])))
# # B = matrix([1., -2., 0., 4.])
# # C = matrix([2., 1.])
#
# sol = solvers.lp(C, A, B)
# u[1,1,1,1] = np.array(sol['x'])[8:12,0]
# print(u.shape)

def indentify_controller(state,u):
    '''
    :param state: current state
    :param u: (n,n,4) tensor
    :return: specific controller
    '''
    x,y,theta,v = state[0] , state[1], state[2], state[ 3]
    if x<-2 or x>2 or y>2 or y<-2:
        return np.zeros_like(state)
    else:
        n = len(u)
        lin_x = np.linspace(-2,2,n)
        lin_y = np.linspace(-2,2,n)
        lin_theta = torch.linspace(-3, 3, n)
        lin_v = torch.linspace(-3, 3, n)
        x_index = np.where(lin_x<x)[0][-1]
        y_index = np.where(lin_y<y)[0][-1]
        theta_index = np.where(lin_theta<theta)[0][-1]
        v_index = np.where(lin_v<v)[0][-1]
        return u[x_index,y_index,theta_index,v_index]


def crown(model,n):
    start = timeit.default_timer()
    lb_tensor = torch.zeros([n, n, n, n, 4])
    ub_tensor = torch.zeros([n,n,n,n,4])
    lin_x = torch.linspace(-2, 2, n)
    lin_y = torch.linspace(-2, 2, n)
    lin_theta = torch.linspace(-3, 3, n)
    lin_v = torch.linspace(-3, 3, n)
    for i in tqdm(range(n)):
        for j in range(n):
            for k in range(n):
                for l in range(n):
                    my_input = torch.tensor([[lin_x[i]+4/n,lin_y[j]+4/n,lin_theta[k]+6/n,lin_v[l]+6/n]]) # midpoint at the sub-interval
                    model = BoundedModule(model, my_input)
                    # Define perturbation. Here we add Linf perturbation to input data.
                    ptb = PerturbationLpNorm(norm=np.inf, eps=0.1)
                    # Make the input a BoundedTensor with the pre-defined perturbation.
                    my_input = BoundedTensor(my_input, ptb)
                    # Regular forward propagation using BoundedTensor works as usual.
                    prediction = model(my_input)
                    # Compute LiRPA bounds using the backward mode bound propagation (CROWN).
                    lb, ub = model.compute_bounds(x=(my_input,), method="backward")
                    lb_tensor[i,j,k,l,:] = lb[0]
                    ub_tensor[i,j,k,l,:] = ub[0]
    stop = timeit.default_timer()
    print(f'done! Total time:{stop-start}, overall cycles: {n**4}')
    return lb_tensor,ub_tensor



class MyModel(nn.Module):
    def __init__(self, n_input, n_hidden, n_output,dt):
        super(MyModel, self).__init__()
        self.dt = dt
        self.model = NormalLyapunovFunction(n_input,n_hidden,n_output,(D_in,),0.1,[12,12,1],eps)
        self.model._control.load_state_dict(torch.load('./data/control_normal_0.5.pkl'))

    def drift(self,state):
        x, y, theta, v = state[:, 0:1], state[:, 1:2], state[:, 2:3], state[:, 3:4]
        dx = v * torch.cos(theta)
        dy = v * torch.sin(theta)
        dtheta = v
        dv = x ** 2 + y ** 2
        ds = torch.cat((dx, dy, dtheta, dv), dim=1)
        return ds

    def forward(self, x):
        # Define your computation here.
        x_new = x + self.dt * (self.drift(x)+self.model._control(x)-self.model._control(torch.zeros_like(x)))
        # x_new = x + self.dt * (self.drift(x)+self.model._control(x)*x)
        return x_new


D_in = 4
H1 = 12
D_out = 4
eps = 0.001
# model = NormalLyapunovFunction(D_in,H1,D_out,(D_in,),0.1,[12,12,1],eps)
# model._icnn.load_state_dict(torch.load('./data/V_normal_0.5.pkl'))
# model._control.load_state_dict(torch.load('./data/control_normal_0.5.pkl'))
start = timeit.default_timer()
forward_model = MyModel(D_in,H1,D_out,0.01)
num = 10
lb, ub = crown(forward_model,num)
print(lb.shape)
u = lq(lb,ub)
stop = timeit.default_timer()
print('total time:',stop-start)
# np.save('./data/cbf1_crown_{}'.format(num),{'lb':lb.detach().numpy(),'ub':ub.detach().numpy()})
# np.save('./data/cbf1_controller_{}'.format(num),u)
print(u.shape)