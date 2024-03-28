import random

import numpy as np
import timeit
import matplotlib.pyplot as plt
import torch
from scipy.integrate import odeint
import networkx as nx
from time import time


def drift(state):
    g = 9.81
    L1,L2=1.,1.
    m1,m2=1.,1.
    theta1, z1, theta2, z2 = state
    ds = np.zeros_like(state)
    c, s = np.cos(theta1-theta2), np.sin(theta1-theta2)
    theta1dot = z1
    z1dot = (m2*g*np.sin(theta2)*c - m2*s*(L1*z1**2*c + L2*z2**2) -
             (m1+m2)*g*np.sin(theta1)) / L1 / (m1 + m2*s**2)
    theta2dot = z2
    z2dot = ((m1+m2)*(L1*z1**2*s - g*np.sin(theta2) + g*np.sin(theta1)*c) +
             m2*L2*z2**2*s*c) / L2 / (m1 + m2*s**2)
    ds = np.array([theta1dot,z1dot,theta2dot,z2dot])
    return ds

def diff(state):
    k = 1.
    theta1, z1, theta2, z2 = state
    ds = np.zeros_like(state)
    ds[1]=np.sin(theta1)*k
    ds[3]=np.sin(theta2)*k
    return ds

def trans(state):
    L1,L2=1.,1.
    theta1,theta2 = state[:,0],state[:,2]
    x1 = L1 * np.sin(theta1)
    y1 = -L1 * np.cos(theta1)
    x2 = x1 + L2 * np.sin(theta2)
    y2 = y1 - L2 * np.cos(theta2)
    coord = np.zeros_like(state)
    coord[:,0],coord[:,1],coord[:,2],coord[:,3]=x1,y1,x2,y2
    return coord

def ZBF(state):
    lower_bound = -0.5
    return np.sin(state[:,0])-lower_bound



'''
visualizing
'''
if __name__ == "__main__": #运行任务
# if __name__ != "__main__": #终止运行，可用于函数测试阶段
    start_time=time()
    '''
    params
    '''
    T = 20  # time
    dt = 0.01 # euler step
    N = int(T/dt) # calculate the numbers

    '''
    initialize
    '''


    x0 = np.random.uniform(-np.pi/3,4*np.pi/3,4) # initial value
    # x0 = np.array([np.pi,0.,np.pi,0.])
    X = np.zeros([N,4])
    X[0,:]=x0
    np.random.seed(23) # our lucky number: 0,1,3,9,23
    noise = np.random.normal(0,1,N)
    '''
    generate
    '''
    for i in range(N-1):  # uncontrolled trajectories with coupling
        x = X[i]
        # new_x = x+dt*drift(x)
        new_x = x+dt*drift(x)+np.sqrt(dt)*diff(x)*noise[i]
        X[i+1,:]=new_x
    # X = trans(X)
    # X = np.mod(X,2*np.pi)

    '''
    plot
    '''
    plt.plot(np.arange(N) * dt, np.sin(X[:, 0])+0.5)
    # plt.plot(np.arange(N) * dt, X[:, 2])
    # plt.axhline(np.pi,ls='--')
    plt.ylim(-1,2)
    # fig, ax = plt.subplots()
    # ax.cla()
    # for i in range(N):
    #     plt.scatter(X[i,0],X[i,1],color='r')
    #     plt.scatter(X[i,2], X[i, 3], color='b')
        # plt.plot(np.arange(N)*dt,X[:,2*i]) # time trajectories
        # plt.pause(0.1)
    print('Run Time:',time()-start_time)
    plt.show()