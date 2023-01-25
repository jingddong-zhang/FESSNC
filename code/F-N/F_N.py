import random

import numpy as np
import timeit
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import networkx as nx
from time import time




def f(state,epi):
    '''
    params
    '''
    beta = 0.7
    mu = 0.8
    I = 1.
    vc,v0,vth=2.8,1.,0.1
    g = 0.4
    N = int(len(state))
    num = int(N/2)

    '''
    根据F-N模型表达式生成不加控制下的向量场
    '''
    v = state[0:N:2] #获得变量v
    w = state[1:N:2] #获得变量w
    ds = np.zeros_like(state)
    # 向量场赋值
    ds[0:N:2] = v - v ** 3 / 3 - w + I
    ds[1:N:2] = epi*(v+beta-mu*w)

    return ds

def g(state,syn_matrix):
    '''
    params
    '''
    vc,v0,vth=2.8,1.,0.1
    k = 1./3.
    N = int(len(state))
    num = int(N/2)

    '''
    根据F-N模型表达式生成不加控制下的向量场
    '''
    v = state[0:N:2] #获得变量v
    w = state[1:N:2] #获得变量w
    # syn_data = (1/(1.+np.exp(-(v)/vth))).reshape(-1,1) #指数型耦合函数
    syn_data = v.reshape(-1, 1)
    syn_state = np.matmul(syn_matrix,syn_data).T[0] #根据连接矩阵得到耦合函数
    ds = np.zeros_like(state)
    # 向量场赋值
    ds[0:N:2] = k*syn_state

    return ds

'''
visualizing
'''
if __name__ == "__main__": #运行任务
# if __name__ != "__main__": #终止运行，可用于函数测试阶段
    start_time=time()
    '''
    params
    '''
    num = 50
    T = 100  # time
    dt = 0.01 # euler step
    N = int(T/dt) # calculate the numbers

    '''
    initialize
    '''

    x0 = np.zeros([2*num])
    x0[0:2*num:2] = np.linspace(-2,2,num)
    x0[1:2*num:2] = np.linspace(0,2,num)
    # x0 = np.random.normal(0,1,2*num) # initial value
    epi = np.random.uniform(0.1,0.1,num)  # unifrom dynamic
    # epi = np.concatenate((np.random.uniform(0.01,0.015,int(num/2)),np.random.uniform(0.025,0.03,int(num/2)))) # two groups of dynamics
    X = np.zeros([N,2*num])
    X[0,:]=x0
    np.random.seed(23) # our lucky number: 0,1,3,9,23
    noise = np.random.normal(0,1,N)
    # syn_matrix = (np.ones(num)-np.eye(num)*(num-1.))/(num-1)  # 连接结构，不分簇版本
    '''
    laplace matrix
    '''
    # ws = nx.watts_strogatz_graph(num, 2, 0.5) #小世界网络
    # syn_matrix = nx.to_numpy_matrix(ws)
    # syn_matrix = syn_matrix - np.diag(np.array(np.sum(syn_matrix,axis=1)).T[0]) # 小世界网络对应的laplace矩阵
    # syn_matrix = np.array(syn_matrix)
    # np.save('./data/laplace_matrix',syn_matrix)
    syn_matrix=np.load('./data/laplace_matrix.npy')
    # nx.draw(ws,pos = nx.circular_layout(ws))
    '''
    generate
    '''
    for i in range(N-1):  # uncontrolled trajectories with coupling
        x = X[i]
        new_x = x+dt*f(x,epi)+np.sqrt(dt)*g(x,syn_matrix)*noise[i]
        X[i+1,:]=new_x

    '''
    process to the difference system
    '''
    # X[:,0:2*num:2] += -X[:,0:1].repeat([50],axis=1)
    # X[:,0:2*num:2] += -X[:,0]
    # print(X[:,0:1].shape)
    '''
    plot
    '''
    # plt.plot(X[:,0],X[:,1]) # phase orbit
    for i in range(num):
        plt.plot(np.arange(N)*dt,X[:,2*i]) # time trajectories

    # plt.ylim(-3.5,3.5)
    print('Run Time:',time()-start_time)
    plt.title('Oscillator Numbers:{}'.format(num))
    plt.show()