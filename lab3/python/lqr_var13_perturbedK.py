import numpy as np
from scipy.linalg import solve_continuous_are, eigvals
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import os

A = np.array([[0.0, 1.0],[4.0, 3.0]])
b = np.array([[2.0],[6.0]])
Q = np.array([[6.0, 0.0],[0.0, 3.0]])
r = 2.0

# Optimal K via algebraic Riccati equation
P = solve_continuous_are(A, b, Q, r)
K_opt = (1.0/r) * (b.T @ P)  # shape (1,2)

# 5% perturbation: first gain +5%, second gain -5%
K_tilde = K_opt.copy()
K_tilde[0, 0] *= 1.05
K_tilde[0, 1] *= 0.95

Acl_opt = A - b @ K_opt
Acl_tilde = A - b @ K_tilde

stable_opt = np.all(np.real(eigvals(Acl_opt)) < 0)
stable_tilde = np.all(np.real(eigvals(Acl_tilde)) < 0)
print({
    'K_opt': K_opt.tolist(),
    'K_tilde': K_tilde.tolist(),
    'stable_opt': bool(stable_opt),
    'stable_tilde': bool(stable_tilde)
})

def lqr_cost(x, u):
    return float(x.T @ Q @ x + r * u**2)

x0 = np.array([1.0, 0.0])
T_span = (0.0, 10.0)

def simulate(K):
    Acl = A - b @ K
    def odefun(t, z):
        x = z[:2]
        u = - float(K @ x)
        jdot = lqr_cost(x, u)
        dx = (Acl @ x)
        return np.hstack([dx, jdot])
    z0 = np.hstack([x0, 0.0])
    sol = solve_ivp(odefun, T_span, z0, max_step=0.01, rtol=1e-8, atol=1e-10)
    t = sol.t
    x = sol.y[:2, :]
    Jt = sol.y[2, :]
    u = - (K @ x).ravel()
    return t, x, u, Jt

t_o, x_o, u_o, J_o = simulate(K_opt)
t_p, x_p, u_p, J_p = simulate(K_tilde)

os.makedirs('/home/leonidas/projects/itmo/optimal-control-theory/lab3/images/task3', exist_ok=True)

plt.figure(figsize=(6,4))
plt.plot(t_o, x_o[0], label='x1 opt')
plt.plot(t_o, x_o[1], label='x2 opt')
plt.plot(t_p, x_p[0], '--', label='x1 pert')
plt.plot(t_p, x_p[1], '--', label='x2 pert')
plt.xlabel('t'); plt.ylabel('states'); plt.grid(True, alpha=0.3)
plt.title('States: optimal vs perturbed K')
plt.legend(); plt.tight_layout()
plt.savefig('/home/leonidas/projects/itmo/optimal-control-theory/lab3/images/task3/states_compare_pertK.png', dpi=200)

plt.figure(figsize=(6,4))
plt.plot(t_o, u_o, label='u opt')
plt.plot(t_p, u_p, '--', label='u pert')
plt.xlabel('t'); plt.ylabel('u'); plt.grid(True, alpha=0.3)
plt.title('Control: optimal vs perturbed K')
plt.legend(); plt.tight_layout()
plt.savefig('/home/leonidas/projects/itmo/optimal-control-theory/lab3/images/task3/u_compare_pertK.png', dpi=200)

plt.figure(figsize=(6,4))
plt.plot(t_o, J_o, label='J opt')
plt.plot(t_p, J_p, '--', label='J pert')
plt.xlabel('t'); plt.ylabel('J(0,t)'); plt.grid(True, alpha=0.3)
plt.title('Accumulated cost: optimal vs perturbed K')
plt.legend(); plt.tight_layout()
plt.savefig('/home/leonidas/projects/itmo/optimal-control-theory/lab3/images/task3/J_compare_pertK.png', dpi=200)

print({'J_inf_opt': float(J_o[-1]), 'J_inf_pert': float(J_p[-1])})


