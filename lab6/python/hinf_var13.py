import os
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import eigvals, solve_continuous_are

A = np.array([[0.0, 1.0], [4.0, 0.0]])
B = np.array([[2.0], [6.0]])
Bf = np.array([[0.0], [2.0]])
Q = np.eye(2)

P0 = solve_continuous_are(A, B, Q, np.eye(1))


def try_solve(gamma: float):
    P = P0.copy()
    try:
        for _ in range(40):
            Qeff = Q + (1.0 / (gamma**2)) * (P @ Bf @ Bf.T @ P)
            P_next = solve_continuous_are(A, B, Qeff, np.eye(1))
            if np.linalg.norm(P_next - P, ord="fro") < 1e-10:
                P = P_next
                break
            P = P_next
        K = -(B.T @ P)
        Acl = A + B @ K
        stable = np.all(np.real(eigvals(Acl)) < -1e-6)
        return True, P, K, Acl, stable
    except Exception:
        return False, None, None, None, False


def find_gamma_min():
    # Simplified approach: try discrete values
    gamma_candidates = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]
    
    for gamma in gamma_candidates:
        ok, P, K, Acl, stable = try_solve(gamma)
        if ok and stable:
            print(f"Found feasible gamma = {gamma}")
            return gamma, P, K, Acl
    
    # If no discrete value works, try simple bisection
    lo, hi = 0.1, 100.0
    for _ in range(20):  # Reduce iterations
        mid = np.sqrt(lo * hi)
        ok, P, K, Acl, stable = try_solve(mid)
        if ok and stable:
            hi = mid
        else:
            lo = mid
    return hi, P, K, Acl


gamma_min, P, K, Acl = find_gamma_min()
print({"gamma_min": float(gamma_min), "K": K.tolist(), "eig(A+BK)": eigvals(Acl)})

x0 = np.array([1.0, 0.0])
T = (0.0, 10.0)


def f_in(t: float) -> float:
    return 10 * np.sin(6 * t) + 5 * np.cos(2 * t) + 4 * np.cos(3 * t) + 3 * np.cos(8 * t)


def odefun(t, x):
    u = (K @ x)[0]  # Fix deprecation warning
    dx = A @ x + (B.flatten() * u) + (Bf.flatten() * f_in(t))
    return dx


sol = solve_ivp(odefun, T, x0, max_step=0.01, rtol=1e-6, atol=1e-8)

t = sol.t
x = sol.y
u = (K @ x).ravel()

os.makedirs("/home/leonidas/projects/itmo/optimal-control-theory/lab6/images/task6", exist_ok=True)

plt.figure(figsize=(6, 4))
plt.plot(t, x[0], label="x1")
plt.plot(t, x[1], label="x2")
plt.xlabel("t"); plt.ylabel("states"); plt.grid(True, alpha=0.3)
plt.legend(); plt.tight_layout()
plt.savefig("/home/leonidas/projects/itmo/optimal-control-theory/lab6/images/task6/states.png", dpi=200)

plt.figure(figsize=(6, 4))
plt.plot(t, u)
plt.xlabel("t"); plt.ylabel("u"); plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("/home/leonidas/projects/itmo/optimal-control-theory/lab6/images/task6/u.png", dpi=200)

om = np.logspace(-2, 2, 400)
C1 = np.array([[1.0, 0.0]])
C2 = np.array([[0.0, 1.0]])

sig1, sig2, sig_full, sigu = [], [], [], []
for w in om:
    sI = 1j * w * np.eye(2)
    s_res = np.linalg.inv(sI - Acl)
    Gf = s_res @ Bf
    Gb = s_res @ B
    sig1.append(float(np.abs((C1 @ Gf).item())))
    sig2.append(float(np.abs((C2 @ Gf).item())))
    sig_full.append(float(np.linalg.norm(Gf)))
    sigu.append(float(np.linalg.norm(Gb)))

sig1 = np.array(sig1)
sig2 = np.array(sig2)
sig_full = np.array(sig_full)
sig_u = np.array(sigu)

plt.figure(figsize=(6, 4))
plt.loglog(om, sig1, label="||C1(sI-Acl)^{-1}Bf||")
plt.loglog(om, sig2, label="||C2(sI-Acl)^{-1}Bf||")
plt.loglog(om, sig_full, label="||(sI-Acl)^{-1}Bf||")
plt.loglog(om, sig_u, label="||(sI-Acl)^{-1}B||")
plt.xlabel("ω"); plt.ylabel("gain"); plt.grid(True, which="both", alpha=0.3)
plt.legend(); plt.tight_layout()
plt.savefig("/home/leonidas/projects/itmo/optimal-control-theory/lab6/images/task6/hinf_gains.png", dpi=200)

metrics: Dict[str, Dict[str, float]] = {
    "C1": {"norm": float(sig1.max()), "omega": float(om[np.argmax(sig1)])},
    "C2": {"norm": float(sig2.max()), "omega": float(om[np.argmax(sig2)])},
    "FullBf": {"norm": float(sig_full.max()), "omega": float(om[np.argmax(sig_full)])},
    "B": {"norm": float(sig_u.max()), "omega": float(om[np.argmax(sig_u)])},
}

print({"hinf_metrics": metrics})
