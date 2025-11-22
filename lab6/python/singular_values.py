"""
Вычисление множества сингулярных чисел для замкнутой системы H∞-регулятора.

Для передаточной функции G(s) = (sI - A_cl)^(-1) B_f замкнутой системы
множество сингулярных чисел определяется как:
    σ(G(jω)) = {σ₁(ω), σ₂(ω), ...} для всех ω ∈ ℝ

где σᵢ(ω) - сингулярные числа матрицы G(jω).
"""

import os
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import eigvals, solve_continuous_are, svd

# Параметры системы (вариант 13)
A = np.array([[0.0, 1.0], [4.0, 0.0]])
B = np.array([[2.0], [6.0]])
Bf = np.array([[0.0], [2.0]])
Q = np.eye(2)

# Начальное приближение из LQR
P0 = solve_continuous_are(A, B, Q, np.eye(1))


def solve_hinf_riccati(gamma: float, max_iter: int = 40, tol: float = 1e-10) -> Tuple[bool, np.ndarray, np.ndarray, np.ndarray]:
    """
    Решает уравнение Риккати для H∞-синтеза при заданном γ.
    
    Returns:
        (success, P, K, Acl) - успех, матрица Риккати, усиление, замкнутая матрица
    """
    P = P0.copy()
    try:
        for _ in range(max_iter):
            Qeff = Q + (1.0 / (gamma**2)) * (P @ Bf @ Bf.T @ P)
            P_next = solve_continuous_are(A, B, Qeff, np.eye(1))
            if np.linalg.norm(P_next - P, ord="fro") < tol:
                P = P_next
                break
            P = P_next
        K = -(B.T @ P)
        Acl = A + B @ K
        stable = np.all(np.real(eigvals(Acl)) < -1e-6)
        return stable, P, K, Acl
    except Exception:
        return False, None, None, None


def find_gamma_min() -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """Находит минимальное γ, при котором система стабилизируема."""
    gamma_candidates = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]
    
    for gamma in gamma_candidates:
        stable, P, K, Acl = solve_hinf_riccati(gamma)
        if stable:
            print(f"Found feasible gamma = {gamma}")
            return gamma, P, K, Acl
    
    # Бинарный поиск
    lo, hi = 0.1, 100.0
    for _ in range(20):
        mid = np.sqrt(lo * hi)
        stable, P, K, Acl = solve_hinf_riccati(mid)
        if stable:
            hi = mid
        else:
            lo = mid
    return hi, P, K, Acl


def compute_singular_values(Acl: np.ndarray, Bf: np.ndarray, omega: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Вычисляет сингулярные числа передаточной функции G(s) = (sI - A_cl)^(-1) B_f
    для всех частот ω.
    
    Args:
        Acl: Замкнутая матрица системы (A + BK)
        Bf: Матрица возмущения
        omega: Массив частот ω ∈ ℝ
        
    Returns:
        (sigma_max, sigma_min) - максимальное и минимальное сингулярные числа для каждой частоты
    """
    n = Acl.shape[0]
    sigma_max = []
    sigma_min = []
    
    for w in omega:
        # Вычисляем передаточную функцию G(jω) = (jωI - A_cl)^(-1) B_f
        sI_minus_Acl = 1j * w * np.eye(n) - Acl
        try:
            G = np.linalg.inv(sI_minus_Acl) @ Bf  # G(jω) - матрица 2×1
            
            # Сингулярные числа матрицы G(jω)
            # Для матрицы размера 2×1 есть только одно ненулевое сингулярное число
            # Вычисляем через SVD: G = U Σ V*
            U, s, Vh = svd(G, full_matrices=False)
            
            # s содержит сингулярные числа в убывающем порядке
            if len(s) > 0:
                sigma_max.append(s[0])  # Максимальное сингулярное число
                sigma_min.append(s[-1] if len(s) > 1 else 0.0)  # Минимальное (может быть 0)
            else:
                sigma_max.append(0.0)
                sigma_min.append(0.0)
        except np.linalg.LinAlgError:
            # Если матрица вырождена, сингулярные числа равны 0
            sigma_max.append(0.0)
            sigma_min.append(0.0)
    
    return np.array(sigma_max), np.array(sigma_min)


def singular_value_function(omega: float, Acl: np.ndarray, Bf: np.ndarray) -> float:
    """
    Математическая функция сингулярного числа передаточной функции.
    
    σ(ω) = σ_max((jωI - A_cl)^(-1) B_f)
    
    Args:
        omega: Частота ω ∈ ℝ
        Acl: Замкнутая матрица системы
        Bf: Матрица возмущения
        
    Returns:
        Максимальное сингулярное число при частоте ω
    """
    n = Acl.shape[0]
    sI_minus_Acl = 1j * omega * np.eye(n) - Acl
    try:
        G = np.linalg.inv(sI_minus_Acl) @ Bf
        U, s, Vh = svd(G, full_matrices=False)
        return float(s[0]) if len(s) > 0 else 0.0
    except np.linalg.LinAlgError:
        return 0.0


def main():
    """Основная функция для вычисления и визуализации множества сингулярных чисел."""
    print("H∞-синтез: вычисление множества сингулярных чисел")
    print("=" * 60)
    
    # 1. Находим минимальное γ и синтезируем регулятор
    gamma_min, P, K, Acl = find_gamma_min()
    print(f"\nМинимальное γ: {gamma_min:.6f}")
    print(f"Матрица усиления K: {K}")
    print(f"Собственные значения A_cl: {eigvals(Acl)}")
    
    # 2. Вычисляем сингулярные числа на сетке частот
    omega = np.logspace(-2, 2, 1000)  # Частоты от 0.01 до 100 рад/с
    sigma_max, sigma_min = compute_singular_values(Acl, Bf, omega)
    
    # 3. Вычисляем H∞-норму (максимум сингулярного числа)
    hinf_norm = np.max(sigma_max)
    omega_max = omega[np.argmax(sigma_max)]
    print(f"\nH∞-норма: ||G||_∞ = {hinf_norm:.6f} при ω = {omega_max:.6f} рад/с")
    
    # 4. Сохраняем результаты
    output_dir = "/home/leonidas/projects/itmo/optimal-control-theory/lab6/images/task6"
    os.makedirs(output_dir, exist_ok=True)
    
    # 5. График множества сингулярных чисел
    plt.figure(figsize=(10, 6))
    plt.semilogx(omega, sigma_max, 'b-', linewidth=2, label='σ_max(ω)')
    if np.any(sigma_min > 1e-10):
        plt.semilogx(omega, sigma_min, 'r--', linewidth=1.5, label='σ_min(ω)')
    plt.axhline(y=hinf_norm, color='g', linestyle=':', linewidth=1.5, 
                label=f'||G||_∞ = {hinf_norm:.3f}')
    plt.axvline(x=omega_max, color='g', linestyle=':', linewidth=1.5, alpha=0.5)
    plt.xlabel('ω, рад/с', fontsize=12)
    plt.ylabel('σ(ω)', fontsize=12)
    plt.title('Множество сингулярных чисел передаточной функции G(s) = (sI - A_cl)⁻¹ B_f', 
              fontsize=12)
    plt.grid(True, alpha=0.3, which='both')
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'singular_values.png'), dpi=200)
    print(f"\nГрафик сохранён: {output_dir}/singular_values.png")
    
    # 6. Сохраняем данные в файл
    data_file = os.path.join(output_dir, 'singular_values_data.txt')
    with open(data_file, 'w') as f:
        f.write("# Множество сингулярных чисел передаточной функции G(s) = (sI - A_cl)⁻¹ B_f\n")
        f.write("# Формат: ω (рад/с) | σ_max(ω) | σ_min(ω)\n")
        f.write(f"# H∞-норма: ||G||_∞ = {hinf_norm:.6f} при ω = {omega_max:.6f} рад/с\n")
        f.write("# γ_min = {:.6f}\n".format(gamma_min))
        f.write("# K = {}\n".format(K.tolist()))
        f.write("#\n")
        for w, s_max, s_min in zip(omega, sigma_max, sigma_min):
            f.write(f"{w:.6e}  {s_max:.6e}  {s_min:.6e}\n")
    print(f"Данные сохранены: {data_file}")
    
    # 7. Выводим математическую формулу
    print("\n" + "=" * 60)
    print("МАТЕМАТИЧЕСКОЕ ПРЕДСТАВЛЕНИЕ МНОЖЕСТВА СИНГУЛЯРНЫХ ЧИСЕЛ:")
    print("=" * 60)
    print("\nДля замкнутой системы с H∞-регулятором:")
    print("  A_cl = A + B K")
    print("  G(s) = (sI - A_cl)⁻¹ B_f")
    print("\nМножество сингулярных чисел определяется как:")
    print("  σ(G) = {σ(ω) : ω ∈ ℝ}")
    print("где")
    print("  σ(ω) = σ_max(G(jω))")
    print("  G(jω) = (jωI - A_cl)⁻¹ B_f")
    print("\nСингулярное число вычисляется через SVD:")
    print("  G(jω) = U(ω) Σ(ω) V*(ω)")
    print("  σ(ω) = max(Σ(ω))")
    print(f"\nH∞-норма (максимум множества):")
    print(f"  ||G||_∞ = max_{{ω∈ℝ}} σ(ω) = {hinf_norm:.6f}")
    print(f"  достигается при ω = {omega_max:.6f} рад/с")
    
    return omega, sigma_max, sigma_min, hinf_norm, omega_max


if __name__ == "__main__":
    omega, sigma_max, sigma_min, hinf_norm, omega_max = main()

