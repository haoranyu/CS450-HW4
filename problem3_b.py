from __future__ import division
import numpy as np
import scipy as sp
import numpy.linalg as la
import scipy.optimize as opt
import matplotlib.pyplot as plt

t = np.array([0.0, 0.25, 0.5, 0.75, 1.00, 1.25, 1.5, 1.75, 2.0])
y = np.array([20.0, 51.58, 68.73, 75.46, 74.36, 67.09, 54.73, 37.98, 17.28])

def gaussNewton(f, Jac, Res, Nx, x, t, y):
	tol = 10e-14
	i = 0
	error = 1
	while error > tol:
		J = Jac(x, t)
		r = Res(x, t, y)
		error = la.norm(r)
		print r
		print error
		s = la.lstsq(np.copy(J), np.copy(-r))[0]
		x = Nx(x, s)
		if i > 500:
			print "Error: Too many iterations"
			break
		else:
			i += 1
	return x

def model(x, t):
    return x[0] + x[1]*t + x[2]*(t**2) + x[3]*np.exp(x[4]*t)

def residual(x, t, y):
    return model(x, t) - y

def jac(x, t):
    ans = np.empty((9,5))
    ans[:,0] = 1
    ans[:,1] = t
    ans[:,2] = t**2
    ans[:,3] = np.exp(x[4]*t)
    ans[:,4] = x[3]*t*np.exp(x[4]*t)
    return np.copy(ans)

def next_x(x, s):
	return x+s

x0 = np.array([1, 0, 0, 1, 0])
x = gaussNewton(model, jac, residual, next_x, np.copy(x0), np.copy(t), np.copy(y) )