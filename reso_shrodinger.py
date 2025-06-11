import numpy as np
import matplotlib.pyplot as plt

# Constantes
hbar = 1
m = 1
V0 = 4000   # Profondeur du puits
a = 0.2     # Largeur du puits

# Énergies à tester
E_values = np.linspace(1, 10000, 1000)
T_values = []

for E in E_values:
    k1 = np.sqrt(2 * m * (E + V0)) / hbar  # vecteur d’onde dans le puits

    # Formule correcte pour le puits rectangulaire
    sin_term = np.sin(k1 * a)
    numerator = 4 * E * (E + V0)
    denominator = numerator + V0**2 * sin_term**2
    T = numerator / denominator

    T_values.append(T)

# Affichage
plt.figure(figsize=(10, 5))
plt.plot(E_values / V0, T_values)
plt.title("Transmission T(E) pour un puits de potentiel fini")
plt.xlabel("E / V₀")
plt.ylabel("T(E)")
plt.grid(True)
plt.show()
