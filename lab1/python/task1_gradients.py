import numpy as np

H = np.array([[10.0, 3.0], [3.0, 4.0]])
b = np.array([4.0, 1.0])  # grad J = H w + b, где w = [x,u]
w_star = np.linalg.solve(H, -b)  # аналитический минимум


def grad(w: np.ndarray) -> np.ndarray:
    return H @ w + b


def value(w: np.ndarray) -> float:
    # J(w) = 0.5 w^T H w + b^T w - 5 (эквивалентно исходной функции с точностью до константы)
    return 0.5 * w @ (H @ w) + b @ w - 5.0


def run_gd(gamma: float, steps: int = 6) -> np.ndarray:
    w = np.array([0.0, 0.0], dtype=float)
    hist = []
    for k in range(steps):
        hist.append((k, w[0], w[1], value(w)))
        w = w - gamma * grad(w)
    hist.append((steps, w[0], w[1], value(w)))
    return np.array(hist)


if __name__ == "__main__":
    for gamma in [0.12, 0.05]:
        Hs = run_gd(gamma)
        print(f"gamma={gamma}")
        for row in Hs:
            k, x, u, J = row
            print(int(k), x, u, J)
        print("=> distance to optimum:", np.linalg.norm(Hs[-1, 1:3] - w_star)) 