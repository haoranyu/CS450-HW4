import numpy as np
import scipy as sp
import numpy.linalg as la
import matplotlib.pyplot as plt

t = np.array([0.0, 0.25, 0.5, 0.75, 1.00, 1.25, 1.5, 1.75, 2.0])
y = np.array([20.0, 51.58, 68.73, 75.46, 74.36, 67.09, 54.73, 37.98, 17.28])

tol = 10e-14

np.random.seed(200)
index = np.random.random_integers(0, len(x))
x0 = x[index] + 0.01 * np.random.rand()
y0 = y[index] + 0.01 * np.random.rand()
R = la.norm((x0, y0))

# Gauss-Newton iteration
error = 1.
it = 0
while error > tol:
	d = map(la.norm, zip(x - x0, y - y0))
	J = np.array([np.array([(x0 - xi)/di, (y0 - yi)/di, -1])
		for xi, yi, di in zip(x, y, d)])
	r = np.array([la.norm((xi - x0, yi - y0)) - R
		for xi, yi in zip(x, y)])
	b = np.dot(J.T, r); error = la.norm(b)
	s = la.solve(np.dot(J.T, J), -b)
	x0 += s[0]; y0 += s[1]; R += s[2]
	it += 1
print("Converged to tolerance in %d iterations"%it)

