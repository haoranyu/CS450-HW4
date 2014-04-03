from __future__ import division
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt

n_arr = (np.arange(20) + 1) * 5;

# node generator
mesh_ac = np.linspace(-1, 1, n)

mesh_bd = np.ones(n)
for i in range(n):
	mesh_bd[i] = mesh_bd[i] * np.cos((2 * (i+1) -1) / (2 * n) *  np.pi)

# a)
V_a = np.array([mesh_ac**i for i in range(n)]).T 
print V_a

# b)
V_b = np.array([mesh_bd**i for i in range(n)]).T 
print V_b

# c)
V_c = np.array([np.cos(i * np.arccos(mesh_ac)) for i in range(n)]).T 
print V_c

# d)
V_d = np.array([np.cos(i * np.arccos(mesh_bd)) for i in range(n)]).T 
print V_d