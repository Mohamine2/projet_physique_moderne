import numpy as np
from scipy.linalg import eigh_tridiagonal

"2e algo pour les états stationnaires"
def etats_stationnaires(V, dx, n_etats=5):
    """
    Calcule les n premiers états stationnaires pour un potentiel V(x)
    avec une méthode de discrétisation et résolution de valeurs propres.
    """
    nx = len(V)

    #paramètre dans la formule
    h = 1 #constante planck
    m = 1 #masse

    #discrétisation
    diag = 1 / dx**2 + V
    hors_diag = -0.5 / dx**2 * np.ones(nx - 1)

    # résolution du problème aux valeurs propres
    energies, vecteurs_propres = eigh_tridiagonal(diag, hors_diag)

    return energies[:n_etats], vecteurs_propres[:, :n_etats]
