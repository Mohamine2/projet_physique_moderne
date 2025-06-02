#Algo devant retourner la densité de probabilité de présence de la particule à différents instants

import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation


#f = fonction d'onde initiale  V|(x,0)
#v = potentiel V(x)
#dx = pas d'espace
#dt = pas de temps
#nt = durée en fonction du nombre iteration temporelle

"1er algo pour equation différentielle"
"propagation d'un paquet d'onde"
def proba_particule(f, V, dx, dt, nt):

    #creation tableau densité pour |(V|(x,t))|
    nx = len(f)
    psi = f.copy()
    densities = np.zeros((int(nt/1000)+1, nx), dtype=float)

    h = 1  #(h = h barre) = constante de planck
    m = 1  #masse de la particule
    s = dt / dx**2 #?

    laplacien = np.zeros(nx, dtype=complex)



    frame = 0
    for t in range(nt):

        # dérivé seconde de x (laplacien discret)
        laplacien[1:-1] = psi[2:] - 2 * psi[1:-1] + psi[:-2]

        # equation de Schrodinger(evolution de Schrodinger (methode Euler)) 
        psi[1:-1] += (1j * h / (2 * m)) * s * laplacien[1:-1] - 1j * V[1:-1] * psi[1:-1] * dt / h

        # on enregistre la densité toutes les 1000 itérations
        if t % 1000 == 0:
            densities[frame, :] = np.abs(psi) ** 2
            frame += 1

    transmission_zone = psi[int(1.5 / dx):]  # après le puits
    transmission = np.sum(np.abs(transmission_zone)**2) * dx
    return densities, transmission

"algo 3 pour résoudre analytiquement"
"avec solution de Schroginder"
def resolution_schrodinger(E, V0, a):
    # a = largeur du puits
    # V0 < 0 (puits)
    #hbar = 1
    m = 1
    if E > 0 and V0 < 0:
        k = np.sqrt(2 * m * E)
        kappa = np.sqrt(2 * m * (E - V0))
        numerator = 4 * k**2 * kappa**2
        denominator = (k**2 + kappa**2)**2 * np.sin(kappa * a)**2 + 4 * k**2 * kappa**2
        T = numerator / denominator
        return T
    else:
        return 0

def effet_ramsauer(V_potential, dx, dt, nt, x_array, v0):
    E_values = np.linspace(1, 30, 50)
    T_numerique = []

    sigma = 0.05
    xc = 0.9

    for E in E_values:
        k = math.sqrt(2 * E)
        normalisation = 1 / (math.sqrt(sigma * math.sqrt(math.pi)))
        wp_gauss = normalisation * np.exp(1j * k * x_array - ((x_array - xc) ** 2) / (2 * (sigma ** 2)))
        
        _, T = proba_particule(wp_gauss, V_potential, dx, dt, nt)
        T_numerique.append(T)

    return E_values, T_numerique
