import numpy as np
import matplotlib.pyplot as plt

# Constantes
hbar = 1
m = 1
V0 = 4000   # Attention : V0 > 0 ici car c’est une profondeur
a = 0.2     # Largeur du puits

# Énergies à tester
E_values = np.linspace(1, 10000, 1000)
T_values = []

for E in E_values:
    k = np.sqrt(2 * m * E) / hbar
    k1 = np.sqrt(2 * m * (E + V0)) / hbar

    # Calcul des coefficients analytiques
    M11 = np.exp(-1j * k * a)
    M12 = np.exp(1j * k * a)
    denom = (k1**2 + k**2) * np.sin(k1 * a) + 2j * k * k1 * np.cos(k1 * a)

    # Transmission amplitude F/A (normalisation A = 1)
    F = 4 * k * k1 * np.exp(-1j * k * a) / denom

    T = np.abs(F)**2  # Probabilité de transmission
    T_values.append(T)

# Affichage
plt.figure(figsize=(10, 5))
plt.plot(E_values / V0, T_values)
plt.title("Transmission T(E) pour un puits de potentiel fini")
plt.xlabel("E / V₀")
plt.ylabel("T(E)")
plt.grid(True)
plt.show()
