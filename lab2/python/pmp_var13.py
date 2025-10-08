import numpy as np
from numpy.linalg import solve
import matplotlib.pyplot as plt

T0, T1 = 0.0, 1.0

def psi1(t, A, B):
    return A * np.cos(t) + B * np.sin(t)

def psi2(t, A, B):
    return -A * np.sin(t) + B * np.cos(t)

def x1(t, A, B, C, D):
    return (
        C * np.cos(t) + D * np.sin(t)
        + 0.5 * (-A/2.0 * t * np.cos(t) + B/2.0 * t * np.sin(t))
    )

def x2(t, A, B, C, D):
    return (
        -C * np.sin(t) + D * np.cos(t)
        + 0.5 * (
            -A/2.0 * (np.cos(t) - t * np.sin(t))
            + B/2.0 * (np.sin(t) + t * np.cos(t))
        )
    )

# Build linear system for A,B,C, D from boundary conditions
M0 = np.array([
    [0.0, 0.0, 1.0, 0.0],       # x1(0)=0 -> C=0
    [-0.25, 0.0, 0.0, 1.0],     # x2(0)=0 -> D - A/4 = 0
])
b0 = np.array([0.0, 0.0])

c, s = np.cos(1.0), np.sin(1.0)
row1 = [0.5 * (-1.0/2.0) * (1.0 * c), 0.5 * (1.0/2.0) * (1.0 * s), c, s]
row2 = [0.5 * (-1.0/2.0) * (c - 1.0 * s), 0.5 * (1.0/2.0) * (s + 1.0 * c), -s, c]
M1 = np.array([row1, row2])
b1 = np.array([2.0, 0.0])

M = np.vstack([M0, M1])
b = np.hstack([b0, b1])
A, B, C, D = solve(M, b)

N = 400
T = np.linspace(T0, T1, N)
PSI2 = psi2(T, A, B)
U = 0.5 * PSI2
X1 = x1(T, A, B, C, D)
X2 = x2(T, A, B, C, D)

print({
    "A": float(A), "B": float(B), "C": float(C), "D": float(D),
    "x1(0)": float(X1[0]), "x2(0)": float(X2[0]), "x1(1)": float(X1[-1]), "x2(1)": float(X2[-1])
})

J = float(np.trapz(U**2, T))
print({"J": J})

import os
os.makedirs("/home/leonidas/projects/itmo/optimal-control-theory/lab2/images/task2", exist_ok=True)

plt.figure(figsize=(6,4))
plt.plot(T, U, label="u*(t)")
plt.xlabel("t"); plt.ylabel("u")
plt.title("Optimal control u*(t)")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("/home/leonidas/projects/itmo/optimal-control-theory/lab2/images/task2/u_opt.png", dpi=200)

plt.figure(figsize=(6,4))
plt.plot(T, X1, label="x1(t)")
plt.plot(T, X2, label="x2(t)")
plt.xlabel("t"); plt.ylabel("states")
plt.title("Optimal trajectory x*(t)")
plt.legend(); plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("/home/leonidas/projects/itmo/optimal-control-theory/lab2/images/task2/x_opt.png", dpi=200)
