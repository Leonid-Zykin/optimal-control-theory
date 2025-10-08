import numpy as np
from scipy.linalg import solve_continuous_are
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import os

A = np.array([[0.0, 1.0], [4.0, 3.0]])
b = np.array([[2.0], [6.0]])
Q = np.array([[6.0, 0.0], [0.0, 3.0]])
r = 2.0

P = solve_continuous_are(A, b, Q, r)
K = (1.0 / r) * (b.T @ P)
Acl = A - b @ K

x0 = np.array([1.0, 0.0])
T = (0.0, 10.0)


def cost(x: np.ndarray, u: float) -> float:
    return float(x.T @ Q @ x + r * u**2)


def odefun(t, z):
    x = z[:2]
    u = -float(K @ x)
    jdot = cost(x, u)
    dx = Acl @ x
    return np.hstack([dx, jdot])


z0 = np.hstack([x0, 0.0])
sol = solve_ivp(odefun, T, z0, max_step=0.01, rtol=1e-8, atol=1e-10)

t = sol.t
x = sol.y[:2, :]
Jt = sol.y[2, :]
u = -(K @ x).ravel()

os.makedirs("/home/leonidas/projects/itmo/optimal-control-theory/lab4/images/task4", exist_ok=True)

plt.figure(figsize=(6, 4))
plt.plot(t, x[0], label="x1")
plt.plot(t, x[1], label="x2")
plt.xlabel("t"); plt.ylabel("states"); plt.grid(True, alpha=0.3)
plt.legend(); plt.tight_layout()
plt.savefig("/home/leonidas/projects/itmo/optimal-control-theory/lab4/images/task4/states.png", dpi=200)

plt.figure(figsize=(6, 4))
plt.plot(t, u)
plt.xlabel("t"); plt.ylabel("u"); plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("/home/leonidas/projects/itmo/optimal-control-theory/lab4/images/task4/u.png", dpi=200)

plt.figure(figsize=(6, 4))
plt.plot(t, Jt)
plt.xlabel("t"); plt.ylabel("J(0,t)"); plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("/home/leonidas/projects/itmo/optimal-control-theory/lab4/images/task4/J.png", dpi=200)

print({"K": K.tolist(), "J_final": float(Jt[-1]), "x0TPx0": float(x0 @ (P @ x0))})
