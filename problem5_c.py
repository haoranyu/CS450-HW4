from __future__ import division
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt

plt.rc("text", usetex=True)
plt.rc("font", family="serif")

def plot_init():
	plt.clf()
	plt.xlabel(" n ")
	plt.ylabel("Condition Number")
	plt.title("Plot for Problem5 c)")
	plt.hold(True)

def plot_draw(X, Y, case):
	plt.semilogy(X, Y, label="Case %s )" % case)
	plt.legend(loc="best")

def cond_a(n):
	mesh_ac = np.linspace(-1, 1, n)
	V_a = np.array([mesh_ac**i for i in range(n)]).T 
	return la.cond(V_a)

def cond_b(n):
	mesh_bd = np.ones(n)
	for i in range(n):
		mesh_bd[i] = mesh_bd[i] * np.cos((2 * (i+1) -1) / (2 * n) *  np.pi)
	V_b = np.array([mesh_bd**i for i in range(n)]).T 
	return la.cond(V_b)

def cond_c(n):
	mesh_ac = np.linspace(-1, 1, n)
	V_c = np.array([np.cos(i * np.arccos(mesh_ac)) for i in range(n)]).T 
	return la.cond(V_c)

def cond_d(n):
	mesh_bd = np.ones(n)
	for i in range(n):
		mesh_bd[i] = mesh_bd[i] * np.cos((2 * (i+1) -1) / (2 * n) *  np.pi)
	V_d = np.array([np.cos(i * np.arccos(mesh_bd)) for i in range(n)]).T 
	return la.cond(V_d)

n_arr = (np.arange(20) + 1) * 5;

a_result = np.ones(20)
for i in range(20):
	a_result[i] *= cond_a(np.copy(n_arr[i]))


b_result = np.ones(20)
for i in range(20):
	b_result[i] *= cond_b(np.copy(n_arr[i]))


c_result = np.ones(20)
for i in range(20):
	c_result[i] *= cond_c(np.copy(n_arr[i]))


d_result = np.ones(20)
for i in range(20):
	d_result[i] *= cond_d(np.copy(n_arr[i]))


plot_init()
plot_draw(n_arr, a_result, 'a')
plot_draw(n_arr, b_result, 'b')
plot_draw(n_arr, c_result, 'c')
plot_draw(n_arr, d_result, 'd')
plt.savefig("problem5_c.png")