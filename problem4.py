from __future__ import division
import numpy as np
import scipy as sp
import numpy.linalg as la
import scipy.special as spc
import matplotlib.pyplot as plt

plt.rc("text", usetex=True)
plt.rc("font", family="serif")

def plot_init():
	plt.clf()
	plt.xlabel(" n ")
	plt.ylabel("lebesgue Constant")
	plt.title("Plot for Problem4")
	plt.hold(True)

def plot_draw(X, Y, case):
	plt.semilogy(X, Y, label=" %s " % case)
	plt.legend(loc="best")

def lambdas(x, xn, n):
    lam = 0
    l = np.ones(n)
    for j in range(n):
		for i in range(n):
		   if i != j:
		       l[j] *= (x - xn[i])/(xn[j]-xn[i])
    for basis in l:
		lam +=  abs(basis)
    return lam

def lebe_const(xn, n):
    leb = []
    grid = np.linspace(-1, 1, 2000)
    for pt in grid:
        leb.append(lambdas(pt,xn,n))
    lebConst = max(leb)
    return lebConst

def equispaced(n):
    node = np.linspace(-1, 1, n)
    return lebe_const(node,n)

def chebyshev(n):
    node = np.ones(n)
    for j in range(n):
        node[j] = np.cos((2 * (j + 1) - 1) * np.pi / (2 * n))
    return lebe_const(node,n)

def gauss_legendre(n):
	node = spc.legendre(n).weights[:,0]
   	return lebe_const(node,n)

n_arr = (np.arange(5) + 1) * 5

e_result = np.ones(5)
for i in range(5):
	e_result[i] *= equispaced(np.copy(n_arr[i]))

c_result = np.ones(5)
for i in range(5):
	c_result[i] *= chebyshev(np.copy(n_arr[i]))

g_result = np.ones(5)
for i in range(5):
	g_result[i] *= gauss_legendre(np.copy(n_arr[i]))

plot_init()
plot_draw(n_arr, e_result, 'Equispaced Nodes')
plot_draw(n_arr, c_result, 'Chebyshev Nodes')
plot_draw(n_arr, g_result, 'Gauss-Legendre Nodes')
plt.savefig("problem4.png")