from __future__ import division
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt

def plot_init():
	plt.clf()
	plt.xlabel("k axis")
	plt.ylabel("ri axis")
	plt.title("Plot for Problem2 c)")
	plt.hold(True)
	plt.gca().set_aspect("equal")

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


