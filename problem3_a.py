from __future__ import division
import numpy as np
import matplotlib.pyplot as plt

def steep_desc(x0, A):
	tol = 10e-14
	err = 1.
	k = 0
	xk = [x0]
	while err > tol:
		sk = -dec
		# r = b - np.dot(A,x0)
		# x1 = x0 + np.dot(r.T,r) / np.dot(r.T,np.dot(A,r)) * r
		# err = np.linalg.norm(abs(x1 - x0))
		# xk.append(x1)
		# x0 = x1
		# k = k + 1
	return k, xk

def f(x, y):
	return (x**2 + y - 11)**2 + (x + y**2 - 7)**2

def deltaf(x, y):
	return (x**2 + y - 11)**2 + (x + y**2 - 7)**2