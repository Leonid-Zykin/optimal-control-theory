import os
from dataclasses import dataclass
from typing import Callable, Dict

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import solve_continuous_are

A = np.array([[0.0, 1.0], [-4.0, -7.0]])
b = np.array([[3.0], [2.0]])
C = np.array([[1.0, 0.0]])
G = np.eye(2)
W_nom = np.array([[7.0, 4.5], [4.5, 6.0]])
V_nom = 2.0

x0 = np.array([1.0, 0.0])
xhat0 = np.zeros(2)
T_span = (0.0, 10.0)

output_dir = "/home/leonidas/projects/itmo/optimal-control-theory/lab5/images/task5"
os.makedirs(output_dir, exist_ok=True)


def observer_gain(W: np.ndarray, V: float) -> np.ndarray:
    P = solve_continuous_are(A.T, C.T, G @ W @ G.T, V)
    return P @ C.T / V


def lqr_gain(Q: np.ndarray, r: float) -> np.ndarray:
    P = solve_continuous_are(A, b, Q, r)
    return (1.0 / r) * b.T @ P


def simulate_observer(
    L_gain: np.ndarray,
    name: str,
    u_fun: Callable[[float], float],
    use_estimate_feedback: bool = False,
    K_gain: np.ndarray | None = None,
    W_noise: np.ndarray | None = None,
    V_noise: float | None = None,
    seed: int | None = None,
) -> Dict[str, np.ndarray]:
    if W_noise is None:
        W_noise = W_nom
    if V_noise is None:
        V_noise = V_nom
    
    # Генерация шумов для всего интервала времени
    if seed is not None:
        np.random.seed(seed)
    
    # Разложение Холецкого для генерации коррелированного шума
    L_w = np.linalg.cholesky(W_noise)
    dt_noise = 0.01  # шаг для генерации шума (совпадает с max_step)
    t_noise = np.arange(T_span[0], T_span[1] + dt_noise, dt_noise)
    n_steps = len(t_noise)
    
    # Генерация белого шума процесса (w) и измерений (v)
    # Для непрерывного белого шума: w(t) имеет интенсивность W, масштабируем на sqrt(dt)
    w_white = np.random.randn(n_steps, 2) @ L_w.T  # коррелированный шум процесса
    v_white = np.random.randn(n_steps) * np.sqrt(V_noise)  # шум измерений
    
    def rhs(t, z):
        x = z[:2]
        xh = z[2:4]
        
        # Интерполяция шумов для текущего времени (ближайший сосед)
        idx = int((t - T_span[0]) / dt_noise)
        idx = min(max(idx, 0), n_steps - 1)
        
        # Масштабирование для белого шума в непрерывной модели
        w_t = w_white[idx] / np.sqrt(dt_noise)
        v_t = v_white[idx] / np.sqrt(dt_noise)
        
        # Измерение с шумом
        y = (C @ x).item() + v_t
        
        if use_estimate_feedback and K_gain is not None:
            u_val = -float((K_gain @ xh).item())
        else:
            u_val = float(u_fun(t))
        
        # Динамика с шумом процесса
        dx = A @ x + (b.flatten() * u_val) + (G @ w_t)
        innovation = y - (C @ xh).item()
        dxh = A @ xh + (b.flatten() * u_val) + (L_gain.flatten() * innovation)
        e = x - xh
        jdot = float(e @ e)
        return np.hstack([dx, dxh, jdot])

    z0 = np.hstack([x0, xhat0, 0.0])
    sol = solve_ivp(rhs, T_span, z0, max_step=0.01, rtol=1e-8, atol=1e-10)
    t = sol.t
    x = sol.y[0:2, :]
    xh = sol.y[2:4, :]
    Jt = sol.y[4, :]
    Eh = x - xh
    if use_estimate_feedback and K_gain is not None:
        U = -(K_gain @ xh).squeeze()
    else:
        U = np.sin(t)

    return {
        "t": t,
        "x": x,
        "xh": xh,
        "Eh": Eh,
        "J": Jt,
        "u": U,
        "L": L_gain,
        "name": name,
    }


def settling_time(t: np.ndarray, Eh: np.ndarray, eps: float = 0.02) -> float:
    cond = (np.max(np.abs(Eh), axis=0) < eps)
    for idx in range(len(t)):
        if np.all(cond[idx:]):
            return float(t[idx])
    return float("nan")


def plot_errors(res: Dict[str, np.ndarray], fname: str, title: str):
    t = res["t"]
    Eh = res["Eh"]
    plt.figure(figsize=(6, 4))
    plt.plot(t, Eh[0], label="e_h1")
    plt.plot(t, Eh[1], label="e_h2")
    plt.xlabel("t")
    plt.ylabel("Ошибка")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.title(title)
    plt.savefig(os.path.join(output_dir, fname), dpi=200)


def plot_integral(res: Dict[str, np.ndarray], fname: str, title: str):
    t = res["t"]
    Jt = res["J"]
    plt.figure(figsize=(6, 4))
    plt.plot(t, Jt)
    plt.xlabel("t")
    plt.ylabel("J(0,t)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.title(title)
    plt.savefig(os.path.join(output_dir, fname), dpi=200)


def compute_error_covariance_trajectory(
    L_gain: np.ndarray,
    W_noise: np.ndarray,
    V_noise: float,
    t_span: tuple = (0.0, 10.0),
) -> tuple[np.ndarray, np.ndarray]:
    """
    Вычисляет траекторию ковариационной матрицы ошибки P(t) через решение
    дифференциального уравнения Риккати:
    dP/dt = A P + P A^T + G W G^T - P C^T V^{-1} C P
    
    Возвращает (t, trace_P_t) где trace_P_t = E{||e_h(t)||²}
    """
    # Начальное условие: P(0) = E{e_h(0)e_h(0)^T}
    # При x(0)=[1,0]^T, x̂(0)=[0,0]^T: e_h(0) = [1,0]^T
    P0 = np.array([[1.0, 0.0], [0.0, 0.0]])
    
    def dPdt(t, P_vec):
        """Правая часть дифференциального уравнения для P(t)"""
        P = P_vec.reshape(2, 2)
        # dP/dt = A P + P A^T + G W G^T - P C^T V^{-1} C P
        # P C^T V^{-1} C P = (1/V) * P C^T C P
        dP = (A @ P + P @ A.T + 
              G @ W_noise @ G.T - 
              (1.0 / V_noise) * (P @ C.T @ C @ P))
        return dP.flatten()
    
    # Решаем дифференциальное уравнение
    P0_vec = P0.flatten()
    sol = solve_ivp(dPdt, t_span, P0_vec, max_step=0.01, rtol=1e-8, atol=1e-10)
    
    # Вычисляем trace(P(t)) = E{||e_h(t)||²}
    trace_P_t = np.array([np.trace(P.reshape(2, 2)) for P in sol.y.T])
    
    return sol.t, trace_P_t


def plot_instant_criterion(
    res: Dict[str, np.ndarray], 
    fname: str, 
    title: str,
    W_noise: np.ndarray | None = None,
    V_noise: float | None = None,
):
    """График нормированного математического ожидания критерия качества E{||e_h(t)||²}"""
    L_gain = res["L"]
    
    # Используем переданные W и V или номинальные по умолчанию
    W_used = W_noise if W_noise is not None else W_nom
    V_used = V_noise if V_noise is not None else V_nom
    
    # Вычисляем математическое ожидание через ковариационную матрицу
    t_P, trace_P_t = compute_error_covariance_trajectory(
        L_gain, W_used, V_used, T_span
    )
    
    # Нормируем на начальное значение (trace(P(0)) = 1)
    J_norm = trace_P_t / trace_P_t[0] if trace_P_t[0] > 0 else trace_P_t
    
    plt.figure(figsize=(6, 4))
    plt.plot(t_P, J_norm, linewidth=2)
    plt.xlabel("t")
    plt.ylabel("$E\\{\\|e_h(t)\\|^2\\} / E\\{\\|e_h(0)\\|^2\\}$")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.title(title)
    plt.savefig(os.path.join(output_dir, fname), dpi=200)


results = {}

# 1. Номинальный наблюдатель
L_nom = observer_gain(W_nom, V_nom)
results["nominal"] = simulate_observer(L_nom, "L_nom", lambda t: np.sin(t), W_noise=W_nom, V_noise=V_nom, seed=42)
plot_errors(results["nominal"], "error_nominal.png", "Ошибки наблюдателя (номинальные W,V)")
plot_instant_criterion(results["nominal"], "J_nominal.png", "Критерий качества $\\|e_h(t)\\|^2$ (номинал)", W_nom, V_nom)
plt.figure(figsize=(6, 4))
plt.plot(results["nominal"]["t"], results["nominal"]["u"])
plt.xlabel("t")
plt.ylabel("u = \\sin t")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.title("Возмущающее управление")
plt.savefig(os.path.join(output_dir, "u_input.png"), dpi=200)

# 3. Незначительное отклонение L
scale_vec = np.array([[1.15], [0.85]])
L_pert = L_nom * scale_vec
results["L_pert"] = simulate_observer(L_pert, "L_pert", lambda t: np.sin(t), W_noise=W_nom, V_noise=V_nom, seed=42)
plt.figure(figsize=(6, 4))
for key, label in [("nominal", "номинал"), ("L_pert", "L ( +15% / -15%)")]:
    res = results[key]
    plt.plot(res["t"], np.linalg.norm(res["Eh"], axis=0), label=label)
plt.xlabel("t")
plt.ylabel("\\|e_h\\|")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.title("Норма ошибки при отклонении L")
plt.savefig(os.path.join(output_dir, "error_norm_L.png"), dpi=200)
plt.figure(figsize=(6, 4))
for key, label in [("nominal", "номинал"), ("L_pert", "L (+15%/-15%)")]:
    res = results[key]
    plt.plot(res["t"], res["J"], label=label)
plt.xlabel("t")
plt.ylabel("J(0,t)")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.title("Интеграл ошибки: отклонение L")
plt.savefig(os.path.join(output_dir, "J_compare_L.png"), dpi=200)
# Отдельный график критерия для случая с отклонённым L
plot_instant_criterion(results["L_pert"], "J_L.png", "Критерий качества $\\|e_h(t)\\|^2$ при отклонении L", W_nom, V_nom)

# 4. Изменение W (симметричная матрица >0)
W_mod = np.array([[8.5, 3.8], [3.8, 5.5]])
L_W = observer_gain(W_mod, V_nom)
results["W_mod"] = simulate_observer(L_W, "W_mod", lambda t: np.sin(t), W_noise=W_mod, V_noise=V_nom, seed=42)
plot_errors(results["W_mod"], "error_W.png", "Ошибки при усиленном шуме процесса W")
plot_instant_criterion(results["W_mod"], "J_W.png", "Критерий качества $\\|e_h(t)\\|^2$ при изменённых W", W_mod, V_nom)
# График сравнения для W
plt.figure(figsize=(6, 4))
for key, label in [("nominal", "номинал"), ("W_mod", "W изменён")]:
    res = results[key]
    plt.plot(res["t"], np.linalg.norm(res["Eh"], axis=0), label=label)
plt.xlabel("t")
plt.ylabel("\\|e_h\\|")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.title("Сравнение нормы ошибки: изменение W")
plt.savefig(os.path.join(output_dir, "error_norm_W.png"), dpi=200)
plt.figure(figsize=(6, 4))
for key, label in [("nominal", "номинал"), ("W_mod", "W изменён")]:
    res = results[key]
    plt.plot(res["t"], res["J"], label=label)
plt.xlabel("t")
plt.ylabel("J(0,t)")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.title("Сравнение интеграла ошибки: изменение W")
plt.savefig(os.path.join(output_dir, "J_compare_W.png"), dpi=200)

# 5. Изменение V
V_mod = 3.0
L_V = observer_gain(W_nom, V_mod)
results["V_mod"] = simulate_observer(L_V, "V_mod", lambda t: np.sin(t), W_noise=W_nom, V_noise=V_mod, seed=42)
plot_errors(results["V_mod"], "error_V.png", "Ошибки при увеличенном шуме измерений V")
plot_instant_criterion(results["V_mod"], "J_V.png", "Критерий качества $\\|e_h(t)\\|^2$ при изменённом V", W_nom, V_mod)
# График сравнения для V
plt.figure(figsize=(6, 4))
for key, label in [("nominal", "номинал"), ("V_mod", "V изменён")]:
    res = results[key]
    plt.plot(res["t"], np.linalg.norm(res["Eh"], axis=0), label=label)
plt.xlabel("t")
plt.ylabel("\\|e_h\\|")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.title("Сравнение нормы ошибки: изменение V")
plt.savefig(os.path.join(output_dir, "error_norm_V.png"), dpi=200)
plt.figure(figsize=(6, 4))
for key, label in [("nominal", "номинал"), ("V_mod", "V изменён")]:
    res = results[key]
    plt.plot(res["t"], res["J"], label=label)
plt.xlabel("t")
plt.ylabel("J(0,t)")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.title("Сравнение интеграла ошибки: изменение V")
plt.savefig(os.path.join(output_dir, "J_compare_V.png"), dpi=200)

# 6. ЛКГ: регулятор + фильтр Калмана
Q_reg = np.diag([6.0, 3.0])
r_reg = 2.0
K_lqr = lqr_gain(Q_reg, r_reg)
results["lqg"] = simulate_observer(
    L_nom,
    "LQG",
    lambda t: 0.0,
    use_estimate_feedback=True,
    K_gain=K_lqr,
    W_noise=W_nom,
    V_noise=V_nom,
    seed=42,
)
plt.figure(figsize=(6, 4))
res = results["lqg"]
plt.plot(res["t"], res["x"][0], label="x1")
plt.plot(res["t"], res["xh"][0], "--", label="x̂1")
plt.plot(res["t"], res["x"][1], label="x2")
plt.plot(res["t"], res["xh"][1], "--", label="x̂2")
plt.xlabel("t")
plt.ylabel("Состояния")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.title("Состояния в ЛКГ-замкнутой системе")
plt.savefig(os.path.join(output_dir, "states_LQG.png"), dpi=200)

plt.figure(figsize=(6, 4))
plt.plot(res["t"], res["u"])
plt.xlabel("t")
plt.ylabel("u = -K x̂")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.title("Управление ЛКГ")
plt.savefig(os.path.join(output_dir, "u_LQG.png"), dpi=200)

plt.figure(figsize=(6, 4))
plt.plot(res["t"], np.linalg.norm(res["Eh"], axis=0))
plt.xlabel("t")
plt.ylabel("\\|e_h\\|")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.title("Ошибка наблюдателя в ЛКГ-замыкании")
plt.savefig(os.path.join(output_dir, "error_LQG.png"), dpi=200)
# График критерия качества для ЛКГ
plot_instant_criterion(results["lqg"], "J_LQG.png", "Критерий качества $\\|e_h(t)\\|^2$ в ЛКГ-системе", W_nom, V_nom)


def summarize(res_dict: Dict[str, Dict[str, np.ndarray]]):
    summary_rows = []
    for key, res in res_dict.items():
        J_end = float(res["J"][-1])
        peak = float(np.max(np.linalg.norm(res["Eh"], axis=0)))
        t_settle = settling_time(res["t"], res["Eh"])
        summary_rows.append((key, J_end, peak, t_settle))
    header = "{:>10} | {:>10} | {:>10} | {:>10}".format("case", "J(0,10)", "max||e||", "t_settle")
    print(header)
    print("-" * len(header))
    for row in summary_rows:
        name = row[0]
        print("{:>10} | {:>10.4f} | {:>10.4f} | {:>10.4f}".format(name, row[1], row[2], row[3]))


summarize(results)
