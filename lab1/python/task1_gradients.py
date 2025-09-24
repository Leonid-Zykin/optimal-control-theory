import numpy as np
import matplotlib.pyplot as plt

H = np.array([[10.0, 3.0], [3.0, 4.0]])
b = np.array([4.0, 1.0])  # grad J = H w + b, где w = [x,u]
w_star = np.linalg.solve(H, -b)  # аналитический минимум


def grad(w: np.ndarray) -> np.ndarray:
    return H @ w + b


def value(w: np.ndarray) -> float:
    # J(w) = 0.5 w^T H w + b^T w - 5 (эквивалентно исходной функции с точностью до константы)
    return 0.5 * w @ (H @ w) + b @ w - 5.0


def run_gd(gamma: float, steps: int = 40) -> np.ndarray:
    w = np.array([0.0, 0.0], dtype=float)
    hist = []
    for k in range(steps + 1):
        hist.append((k, w[0], w[1], value(w)))
        w = w - gamma * grad(w)
    return np.array(hist)


def plot_trajectories():
    gammas = [0.12, 0.05]
    labels = ["gamma=0.12 (колебательная)", "gamma=0.05 (апериодическая)"]
    colors = ["tab:red", "tab:blue"]

    for gamma, label, color in zip(gammas, labels, colors):
        hist = run_gd(gamma)
        xs, us, Js = hist[:, 1], hist[:, 2], hist[:, 3]

        # 1) Траектория в пространстве (x,u)
        plt.figure(figsize=(5, 4))
        plt.plot(xs, us, marker="o", ms=3, color=color, lw=1)
        plt.plot([w_star[0]], [w_star[1]], marker="*", ms=10, color="k", label="минимум")
        plt.xlabel("x")
        plt.ylabel("u")
        plt.title(f"Траектория в плоскости (x,u), {label}")
        plt.grid(True, alpha=0.3)
        plt.legend(loc="best")
        plt.tight_layout()
        plt.savefig("/home/leonidas/projects/itmo/optimal-control-theory/lab1/images/task1/gd_traj_" + ("012" if gamma==0.12 else "005") + ".png", dpi=200)
        plt.close()

        # 2) Убывание J по итерациям
        plt.figure(figsize=(5, 4))
        plt.plot(Js, marker="o", ms=3, color=color, lw=1)
        plt.xlabel("итерация")
        plt.ylabel("J(x,u)")
        plt.title(f"Убывание критерия J, {label}")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("/home/leonidas/projects/itmo/optimal-control-theory/lab1/images/task1/gd_J_" + ("012" if gamma==0.12 else "005") + ".png", dpi=200)
        plt.close()


if __name__ == "__main__":
    # Печатаем первые итерации как ранее
    for gamma in [0.12, 0.05]:
        hist = run_gd(gamma, steps=6)
        print(f"gamma={gamma}")
        for row in hist:
            k, x, u, J = row
            print(int(k), x, u, J)
        print("=> distance to optimum:", np.linalg.norm(hist[-1, 1:3] - w_star))

    # Строим и сохраняем графики
    plot_trajectories() 