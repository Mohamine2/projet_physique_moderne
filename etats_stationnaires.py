import numpy as np
from scipy.linalg import eigh_tridiagonal
def etats_stationnaires(V, dx, n_etats=5):
    nx = len(V)
    hbar = 1
    m = 1
    facteur = hbar**2 / (2 * m * dx**2)

    diag = 2 * facteur + V
    hors_diag = -facteur * np.ones(nx - 1)

    energies, vecteurs_propres = eigh_tridiagonal(diag, hors_diag)

    # Normalisation continue
    for i in range(n_etats):
        vecteurs_propres[:, i] /= np.sqrt(np.trapz(np.abs(vecteurs_propres[:, i])**2, dx=dx))

    return energies[:n_etats], vecteurs_propres[:, :n_etats]
