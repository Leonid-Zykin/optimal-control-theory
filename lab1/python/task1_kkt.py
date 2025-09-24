import sympy as sp

x, u, lam = sp.symbols('x u lam', real=True)
J = 5*x**2 + 2*u**2 + 3*x*u + 4*x + u - 5
c = 4*x**2 + u + 1
L = J + lam*c

# Начнём поиск решения рядом с аналитическими оценками
initial_guess = (-0.03, -1.0, 3.0)
sol = sp.nsolve([
    sp.diff(L, x),
    sp.diff(L, u),
    c
], [x, u, lam], initial_guess)

x_star, u_star, lam_star = [sp.N(v, 12) for v in sol]
J_star = sp.N(J.subs({x: x_star, u: u_star}), 12)
print({
    'x': float(x_star),
    'u': float(u_star),
    'lambda': float(lam_star),
    'J': float(J_star)
}) 