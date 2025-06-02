import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
from etats_stationnaires import etats_stationnaires #Appel de la fonction

# === Paramètres physiques et numériques ===
dt = 1E-7
dx = 0.001
nx = int(1/dx)*2
nt = 90000
n_frames = int(nt / 1000) + 1

v0 = -4000
e = 5
E = e * v0
k = math.sqrt(2 * abs(E))

x_array = np.linspace(0, (nx - 1) * dx, nx)
V_potential = np.zeros(nx)
V_potential[int(nx * 0.45):int(nx * 0.55)] = v0  # Puits centré

xc = 0.6
sigma = 0.05
normalisation = 1 / (math.sqrt(sigma * math.sqrt(math.pi)))
wp_gauss = normalisation * np.exp(1j * k * x_array - ((x_array - xc) ** 2) / (2 * sigma**2))


density = np.zeros((nt, nx))
density[0, :] = np.abs(wp_gauss)**2
final_density = np.zeros((n_frames, nx))



# Constantes pour Crank-Nicolson
hbar = 1  # unités naturelles
m = 1
s = hbar * dt / (4 * m * dx**2)

# Matrice diagonale du potentiel
V_diag = V_potential

# Construction des matrices A et B (tridiagonales)
main_diag = (1 + 2j * s + 0.5j * dt * V_diag)
off_diag  = -1j * s * np.ones(nx - 1)

A = diags([off_diag, main_diag, off_diag], offsets=[-1, 0, 1], format='csc')

main_diag_B = (1 - 2j * s - 0.5j * dt * V_diag)
B = diags([1j * s * np.ones(nx - 1), main_diag_B, 1j * s * np.ones(nx - 1)],
          offsets=[-1, 0, 1], format='csc')

# Initialisation du paquet d'onde
psi = wp_gauss.copy()

for t in range(1, nt):
    # Calcul du second membre
    rhs = B @ psi

    # Résolution du système linéaire A ψ^{n+1} = rhs
    psi = spsolve(A, rhs)

    # Sauvegarde de la densité de probabilité
    density[t,:] = np.abs(psi)**2

    if t % 1000 == 0:
        final_density[t // 1000, :] = density[t, :]


# === États stationnaires ===
n_etats = 3
energies, etats = etats_stationnaires(V_potential, dx, n_etats=n_etats)

# Tracé des états stationnaires
plt.figure()
for i in range(n_etats):
    psi = etats[:, i]
    norm = np.max(np.abs(psi)**2)
    plt.plot(x_array, np.abs(psi)**2 / norm + energies[i], label=f"État {i+1}, E={energies[i]:.1f}")
plt.plot(x_array, V_potential, 'k--', label="Potentiel")
plt.title("États stationnaires dans le puits")
plt.xlabel("x")
plt.ylabel("Densité normalisée + Énergie")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()


# === Animation Crank-Nicolson ===
def init():
    line.set_data([], [])
    return line,

def animate(j):
    line.set_data(x_array, final_density[j, :])
    return line,


plot_title = f"E/V₀ = {e}"

fig = plt.figure()
line, = plt.plot([], [])
plt.ylim(-1, 1)
plt.xlim(0, 2)
plt.plot(x_array, V_potential, label="Potentiel")
plt.title(plot_title)
plt.xlabel("x")
plt.ylabel("Densité de probabilité de présence")

ani = animation.FuncAnimation(fig, animate, init_func=init,
                              frames=n_frames, blit=False, interval=100, repeat=False)

import os

# Sauvegarde toujours l'animation
ani.save("animation.mp4", fps=30, dpi=150)
print("✅ Animation sauvegardée sous 'animation.mp4'")

# Affiche uniquement si une interface graphique est dispo
if "DISPLAY" in os.environ:
    plt.show()
else:
    print("ℹ️ Aucun affichage graphique détecté (mode terminal).")