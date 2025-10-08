import numpy as np
from scipy.linalg import solve_continuous_are, eigvals
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import os

A = np.array([[0.0, 1.0], [4.0, 3.0]])
b = np.array([[2.0], [6.0]])
Q = np.array([[6.0, 0.0], [0.0, 3.0]])
r = 2.0

P = solve_continuous_are(A, b, Q, r)
K = (1.0 / r) * (b.T @ P)  # (1x2)
Acl = A - b @ K
print("K=", K)
print("eig(A-bK)=", eigvals(Acl))


def lqr_cost(x: np.ndarray, u: float) -> float:
    return float(x.T @ Q @ x + r * u**2)


x0 = np.array([1.0, 0.0])
T = (0.0, 10.0)


def odefun(t, z):
    x = z[:2]
    u = -float(K @ x)
    jdot = lqr_cost(x, u)
    dx = Acl @ x
    return np.hstack([dx, jdot])


z0 = np.hstack([x0, 0.0])
sol = solve_ivp(odefun, T, z0, max_step=0.01, rtol=1e-8, atol=1e-10)

t = sol.t
x = sol.y[:2, :]
Jt = sol.y[2, :]
u = -(K @ x).ravel()

os.makedirs("/home/leonidas/projects/itmo/optimal-control-theory/lab3/images/task3", exist_ok=True)

plt.figure(figsize=(6, 4))
plt.plot(t, x[0], label="x1")
plt.plot(t, x[1], label="x2")
plt.xlabel("t"); plt.ylabel("states"); plt.grid(True, alpha=0.3)
plt.title("Closed-loop states")
plt.legend(); plt.tight_layout()
plt.savefig("/home/leonidas/projects/itmo/optimal-control-theory/lab3/images/task3/states.png", dpi=200)

plt.figure(figsize=(6, 4))
plt.plot(t, u)
plt.xlabel("t"); plt.ylabel("u"); plt.grid(True, alpha=0.3)
plt.title("Control u(t)")
plt.tight_layout()
plt.savefig("/home/leonidas/projects/itmo/optimal-control-theory/lab3/images/task3/u.png", dpi=200)

plt.figure(figsize=(6, 4))
plt.plot(t, Jt)
plt.xlabel("t"); plt.ylabel("J(0,t)"); plt.grid(True, alpha=0.3)
plt.title("Accumulated cost J")
plt.tight_layout()
plt.savefig("/home/leonidas/projects/itmo/optimal-control-theory/lab3/images/task3/J.png", dpi=200)

print({"J_final": float(Jt[-1])})
