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
K_opt = (1.0 / r) * (b.T @ P)

x0 = np.array([1.0, 0.0])
T = (0.0, 10.0)


def cost(x: np.ndarray, u: float) -> float:
    return float(x.T @ Q @ x + r * u**2)


def simulate(K_current: np.ndarray):
    Acl = A - b @ K_current

    def odefun(t, z):
        x = z[:2]
        u = -(K_current @ x)[0]
        jdot = cost(x, u)
        dx = Acl @ x
        return np.hstack([dx, jdot])

    z0 = np.hstack([x0, 0.0])
    sol = solve_ivp(odefun, T, z0, max_step=0.01, rtol=1e-8, atol=1e-10)
    return sol.t, sol.y[:2, :], -(K_current @ sol.y[:2, :]).ravel(), sol.y[2, :], eigvals(Acl)


cases = {
    "opt": K_opt,
    "minus20": K_opt * np.array([[0.8, 0.8]]),
    "plus20": K_opt * np.array([[1.2, 1.2]]),
}

results = {}
for name, K_case in cases.items():
    t, x, u, Jt, eigs = simulate(K_case)
    results[name] = {
        "K": K_case,
        "t": t,
        "Jt": Jt,
        "eig": eigs,
        "J_final": float(Jt[-1]),
    }
    print(name, {"K": K_case.tolist(), "eig": eigs, "J_final": float(Jt[-1])})

os.makedirs("/home/leonidas/projects/itmo/optimal-control-theory/lab4/images/task4", exist_ok=True)

plt.figure(figsize=(6, 4))
for label, style in [("opt", "-"), ("minus20", "--"), ("plus20", ":")]:
    plt.plot(results[label]["t"], results[label]["Jt"], linestyle=style, label=label)

plt.xlabel("t")
plt.ylabel("J(0,t)")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("/home/leonidas/projects/itmo/optimal-control-theory/lab4/images/task4/J_compare_pertK.png", dpi=200)


