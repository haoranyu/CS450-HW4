from __future__ import division
import numpy as np
import scipy as sp
import numpy.linalg as la
import scipy.optimize as opt
import matplotlib.pyplot as plt

t = np.array([0.0, 0.25, 0.5, 0.75, 1.00, 1.25, 1.5, 1.75, 2.0])
y = np.array([20.0, 51.58, 68.73, 75.46, 74.36, 67.09, 54.73, 37.98, 17.28])

def plot_init():
    plt.clf()
    plt.xlabel("t")
    plt.ylabel("f")
    plt.title("Plot for Problem3 b) ")
    plt.hold(True)

def plot_draw(t, y, x):
	plt.scatter(t, y)
	t_sample = np.linspace(0, 2, 256)
	y_sample = model(x, t_sample)
	plt.plot(t_sample, model(x, t_sample))


def gaussNewton(f, Jac, Res, x, t, y):
	tol = 1e-14
	i = 0
	error = 1
	while error > tol:
		J = Jac(x, t)
		r = Res(x, t, y)
		if i > 0:
			error = la.norm(r - r_prev)
		r_prev = r
		s = la.lstsq(np.copy(J), np.copy(-r))[0]
		if i > 500:
			print "Error: Too many iterations"
			break
		else:
			i += 1

		x = x + s
		if(i <= 5):
			print "x %d is" % i
			print x
	return x, (i-1)

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

x0_1 = np.array([0, 0, 0, 0, 1])
x0_2 = np.array([1, 0, 0, 0, 0])
x0_3 = np.array([1, 0, 0, 1, 0])

x, i = gaussNewton(model, jac, residual, np.copy(x0_1), np.copy(t), np.copy(y) )
plot_init()
plot_draw(t, y, x)
plt.savefig("problem3_b_1.png")
print "=============================="

x, i = gaussNewton(model, jac, residual, np.copy(x0_2), np.copy(t), np.copy(y) )
plot_init()
plot_draw(t, y, x)
plt.savefig("problem3_b_2.png")
print "=============================="

x, i = gaussNewton(model, jac, residual, np.copy(x0_3), np.copy(t), np.copy(y) )
plot_init()
plot_draw(t, y, x)
plt.savefig("problem3_b_3.png")