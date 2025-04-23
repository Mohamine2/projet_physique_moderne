import numpy as np
from scipy.linalg import eigh_tridiagonal

def etats_stationnaires(V, dx, n_etats=5):
    """
    Calcule les n premiers états stationnaires pour un potentiel V(x)
    avec une méthode de discrétisation et résolution de valeurs propres.
    """
    nx = len(V)
    hbar = 1
    m = 1

    diag = 1 / dx**2 + V
    off_diag = -0.5 / dx**2 * np.ones(nx - 1)

    # résolution du problème aux valeurs propres
    energies, vecteurs_propres = eigh_tridiagonal(diag, off_diag)

    return energies[:n_etats], vecteurs_propres[:, :n_etats]
