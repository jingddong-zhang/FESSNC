import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from scipy import linalg


g = 9.81
L1, L2 = 1., 1.
m1, m2 = 1., 1.
A = np.array([[0, 1.0,0,0],[(m1+m2)*g/(m1*L1),0,-m2*g/(m1*L1),0],[0,0,0,1.], [-g*(m1+m2)/(m1*L2),0,g*(m1+m2)/(m1*L2),0]])
Q = np.eye(4)

# aX+Xa^T=q, input : a,q
P = linalg.solve_continuous_lyapunov(A.T, -Q)

print(P)