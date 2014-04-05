import numpy as np
import numpy.linalg as la
import scipy.optimize as opt
import matplotlib.pyplot as plt

def plot_init(x, y, f, i, method):
    plt.clf()
    plt.xlabel("")
    plt.ylabel("")
    plt.title("Plot for Problem3 a) "+method)
    plt.hold(True)
    plt.contour(x,y,f,i)

def plot_draw(result):
    plt.plot(result[:,0], result[:,1],'-kx')

def H(x):
    H = np.ones((2,2))
    H[0][0] = 12 * x[0]**2 + 4 * x[1] - 42
    H[0][1] = 4 * x[0] + 4 * x[1]
    H[1][0] = 4 * x[0] + 4 * x[1]
    H[1][1] = 12 * x[1]**2 + 4 * x[0] - 26
    return H

def f(x):
    return (x[0]**2 + x[1] - 11)**2 + (x[0] + x[1]**2 - 7)**2

def df(x):
    dx = 4 * (x[0]**2 + x[1] - 11) * x[0] + 2 * (x[0] + x[1]**2 - 7)
    dy = 2 * (x[0]**2 + x[1] - 11) + 4 * (x[0] + x[1]**2 - 7) * x[1]
    return np.array([dx,dy])

def steepest_desc(s):
    sol = -df(s)
    a = opt.line_search(f, df, s, sol)[0]
    s_n = s + a * sol
    return s_n

def newton(s):
    sol = la.solve(H(s), -df(s))
    s_n = s + sol
    return s_n
   
def damped_newton(s):
    sol = la.solve(H(s), -df(s))
    a = opt.line_search(f, df, s, sol)[0]
    if(a == None):
        a = 1
    s_n = s + a * sol
    return s_n

def solve(s0, method):
    s = s0
    result = [s]
    while 1:
        s_n = method(s)
        result.append(s_n)
        error = la.norm(s - s_n)
        if error < 1e-14:
            break
        else:
            s = s_n
    return np.asarray(result)   

x, y = np.mgrid[-4.0:4.0:0.01, -4.0:4.0:0.01]
F = f([x, y])

plot_init(x, y, F, 15, "Steepest Descent Method")
plot_draw(solve([2,2],steepest_desc))
plot_draw(solve([2,-1],steepest_desc))
plot_draw(solve([-2,2],steepest_desc))
plot_draw(solve([-2,-2],steepest_desc))
plt.savefig("problem3_a_SteepestDescent.png")

plot_init(x, y, F, 15, "Newton's Method")
plot_draw(solve([2,2],newton))
plot_draw(solve([2,-1],newton))
plot_draw(solve([-2,2],newton))
plot_draw(solve([-2,-2],newton))
plt.savefig("problem3_a_Newton.png")

plot_init(x, y, F, 15, "Damped Newton Method")
plot_draw(solve([2,2],damped_newton))
plot_draw(solve([2,-1],damped_newton))
plot_draw(solve([-2,2],damped_newton))
plot_draw(solve([-2,-2],damped_newton))
plt.savefig("problem3_a_DampedNewton.png")