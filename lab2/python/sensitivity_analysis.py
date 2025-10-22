# lab2/python/sensitivity_analysis.py
import numpy as np
import matplotlib.pyplot as plt

# Optimal parameters
A_opt = -37.8663
B_opt = 23.0598
J_min = 349.0044

# Trigonometric constants
sin1 = np.sin(1.0)
cos1 = np.cos(1.0)
sin1cos1 = sin1 * cos1
sin2_1 = sin1**2

def criterion_deviation(dA, dB):
    """Calculate J for deviations dA, dB from optimal values"""
    A = A_opt + dA
    B = B_opt + dB
    return (1/8) * (A**2 * (1 - sin1cos1) - 2*A*B * sin2_1 + B**2 * (1 + sin1cos1))

# Test different deviation scenarios
scenarios = [
    ("No deviation", 0.0, 0.0),
    ("Small A deviation", 1.0, 0.0),
    ("Small B deviation", 0.0, 1.0),
    ("Both small", 1.0, 1.0),
    ("Large A deviation", 5.0, 0.0),
    ("Large B deviation", 0.0, 5.0),
    ("Both large", 5.0, 5.0),
    ("Opposite A", -2.0, 0.0),
    ("Opposite B", 0.0, -2.0)
]

print("Parameter sensitivity analysis:")
print("Scenario\t\t\tJ value\t\tIncrease")
print("-" * 50)
for name, dA, dB in scenarios:
    J = criterion_deviation(dA, dB)
    increase = J - J_min
    print(f"{name:<20}\t{J:.2f}\t\t{increase:.2f}")

# Plot sensitivity surface
dA_range = np.linspace(-5, 5, 21)
dB_range = np.linspace(-5, 5, 21)
DA, DB = np.meshgrid(dA_range, dB_range)
J_surface = np.zeros_like(DA)

for i in range(len(dA_range)):
    for j in range(len(dB_range)):
        J_surface[j, i] = criterion_deviation(dA_range[i], dB_range[j])

plt.figure(figsize=(8, 6))
contour = plt.contour(DA, DB, J_surface, levels=20)
plt.clabel(contour, inline=True, fontsize=8)
plt.plot(0, 0, 'r*', markersize=15, label='Optimal point')
plt.xlabel('Deviation ΔA')
plt.ylabel('Deviation ΔB')
plt.title('Criterion sensitivity to parameter deviations')
plt.colorbar(contour, label='J value')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig('/home/leonidas/projects/itmo/optimal-control-theory/lab2/images/task2/sensitivity.png', dpi=200)

