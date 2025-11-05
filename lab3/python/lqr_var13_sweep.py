import numpy as np
from scipy.linalg import solve_continuous_are, eigvals
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import os

A = np.array([[0.0, 1.0], [4.0, 3.0]])
b = np.array([[2.0], [6.0]])
Q_base = np.array([[6.0, 0.0], [0.0, 3.0]])
x0 = np.array([1.0, 0.0])

def simulate(r: float, Q: np.ndarray, T=(0.0, 10.0)):
    P = solve_continuous_are(A, b, Q, r)
    K = (1.0 / r) * (b.T @ P)
    Acl = A - b @ K

    def odefun(t, z):
        x = z[:2]
        u = -float(K @ x)
        jdot = float(x.T @ Q @ x + r * u * u)
        dx = Acl @ x
        return np.hstack([dx, jdot])

    z0 = np.hstack([x0, 0.0])
    sol = solve_ivp(odefun, T, z0, max_step=0.01, rtol=1e-8, atol=1e-10)
    t = sol.t
    x = sol.y[:2, :]
    Jt = sol.y[2, :]
    u = -(K @ x).ravel()
    return dict(P=P, K=K, Acl=Acl, t=t, x=x, u=u, Jt=Jt,
                J_final=float(Jt[-1]), eig= eigvals(Acl))


def sweep_r_values(r_values=(0.5, 2.0, 5.0)):
    os.makedirs('/home/leonidas/projects/itmo/optimal-control-theory/lab3/images/task3', exist_ok=True)
    plt.figure(figsize=(6, 4))
    for r in r_values:
        res = simulate(r, Q_base)
        plt.plot(res['t'], res['Jt'], label=f"r={r}")
    plt.xlabel('t'); plt.ylabel('J(0,t)'); plt.grid(True, alpha=0.3)
    plt.title('J(0,t) при разных r')
    plt.legend(); plt.tight_layout()
    plt.savefig('/home/leonidas/projects/itmo/optimal-control-theory/lab3/images/task3/J_r.png', dpi=200)


def sweep_k_values(k_values=(0.5, 1.0, 2.0)):
    os.makedirs('/home/leonidas/projects/itmo/optimal-control-theory/lab3/images/task3', exist_ok=True)
    plt.figure(figsize=(6, 4))
    for k in k_values:
        Qk = k * Q_base
        res = simulate(2.0, Qk)
        plt.plot(res['t'], res['Jt'], label=f"k={k}")
    plt.xlabel('t'); plt.ylabel('J(0,t)'); plt.grid(True, alpha=0.3)
    plt.title('J(0,t) при Q_k = k Q')
    plt.legend(); plt.tight_layout()
    plt.savefig('/home/leonidas/projects/itmo/optimal-control-theory/lab3/images/task3/J_k.png', dpi=200)


if __name__ == "__main__":
    sweep_r_values()
    sweep_k_values()
    # Печать сводки по значениям J(∞)
    print("Summary r:")
    for r in (0.5, 2.0, 5.0):
        res = simulate(r, Q_base)
        print({"r": r, "K": res['K'].tolist(), "eig": res['eig'], "J_inf": res['J_final']})
    print("Summary k:")
    for k in (0.5, 1.0, 2.0):
        res = simulate(2.0, k * Q_base)
        print({"k": k, "K": res['K'].tolist(), "eig": res['eig'], "J_inf": res['J_final']})


