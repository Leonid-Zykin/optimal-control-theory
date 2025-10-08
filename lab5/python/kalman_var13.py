import numpy as np
from scipy.linalg import solve_continuous_are
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import os

A = np.array([[0.0, 1.0], [-4.0, -7.0]])
b = np.array([[3.0], [2.0]])
C = np.array([[1.0, 0.0]])
G = np.eye(2)
W = np.array([[7.0, 4.5], [4.5, 6.0]])
V = 2.0

# CARE for observer
P = solve_continuous_are(A.T, C.T, G @ W @ G.T, V)
L = P @ C.T / V
print("L=", L.ravel())

x0 = np.array([1.0, 0.0])
xhat0 = np.zeros(2)
T = (0.0, 10.0)

def u(t: float) -> float:
    return np.sin(t)


def odefun(t, z):
    x = z[:2]
    xh = z[2:4]
    y = C @ x
    ut = u(t)
    dx = A @ x + (b.flatten() * ut)
    dxh = A @ xh + (b.flatten() * ut) + (L @ (y - C @ xh)).flatten()
    e = x - xh
    jdot = float(e @ e)
    return np.hstack([dx, dxh, jdot])


z0 = np.hstack([x0, xhat0, 0.0])
sol = solve_ivp(odefun, T, z0, max_step=0.01, rtol=1e-8, atol=1e-10)

t = sol.t
x = sol.y[0:2, :]
xh = sol.y[2:4, :]
Jt = sol.y[4, :]
Eh = x - xh
U = np.sin(t)

os.makedirs("/home/leonidas/projects/itmo/optimal-control-theory/lab5/images/task5", exist_ok=True)

plt.figure(figsize=(6, 4))
plt.plot(t, Eh[0], label="e_h1")
plt.plot(t, Eh[1], label="e_h2")
plt.xlabel("t"); plt.ylabel("error"); plt.grid(True, alpha=0.3)
plt.legend(); plt.tight_layout()
plt.savefig("/home/leonidas/projects/itmo/optimal-control-theory/lab5/images/task5/error.png", dpi=200)

plt.figure(figsize=(6, 4))
plt.plot(t, U)
plt.xlabel("t"); plt.ylabel("u=sin t"); plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("/home/leonidas/projects/itmo/optimal-control-theory/lab5/images/task5/u.png", dpi=200)

plt.figure(figsize=(6, 4))
plt.plot(t, Jt)
plt.xlabel("t"); plt.ylabel("J(0,t)"); plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("/home/leonidas/projects/itmo/optimal-control-theory/lab5/images/task5/J.png", dpi=200)
