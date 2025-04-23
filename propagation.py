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

def proba_particule(f, V, dx, dt, nt):

    #creation tableau densité pour |(V|(x,t))|
    nx = len(f)
    psi = f.copy()
    densities = np.zeros((int(nt/1000)+1, nx), dtype=float)

    h = 1  #(h = h barre) = constante de planck
    m = 1  #masse de la particule
    s = dt / dx**2

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


