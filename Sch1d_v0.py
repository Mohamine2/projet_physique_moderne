import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from propagation import proba_particule
from propagation import resolution_schrodinger
from propagation import effet_ramsauer
from etat_stationnaire import etats_stationnaires

#initialisation des animations
def init():
    line.set_data([], [])
    return line,

def animate(j):
    line.set_data(x_array, final_density[j,:]) #Crée un graphique pour chaque densite sauvegarde
    return line,

#paramètres physique
#les constantes données || #v0=-4000 #e=5(E/V0) # E=e*v0 #nt=90000
dt=1E-7 #pas temps
dx=0.001 # pas d'espace
nx=int(1/dx)*2 # nombre point spatiaux
nt=500 # nombre de pas temorels (# En fonction du potentiel il faut modifier ce parametre car sur certaines animations la particule atteins les bords)
n_frames=int(nt/1000)+1#nombre d image dans notre animation
s=dt/(dx**2) #?

"valeur qu'on peut changer pour test"
v0=9 #profondeur du puits
E=5  #energie de la particule
e=E/v0 # E/v0
k=math.sqrt(2*abs(E)) #nombre d'onde

print(f"E = {E}, correspondant à e = {e} et v0 = {v0}") #affiche les valeurs choisit

x_array = np.linspace(0, (nx - 1) * dx, nx)

#potenciel
V_potential = np.zeros(nx) #potenciel nul partout dcp initialiser
for i in range(nx):
    if 0.8 <= x_array[i] <= 1.0:  # zone du puits de potentiel
        V_potential[i] = -v0

"Paquet ondes gaussien"
#gaussian wave packet (Paquet ondes gaussien)
xc=0.9 #initial à 0.6 #position initiale (centre)
sigma=0.05 #ecart type
normalisation=1/(math.sqrt(sigma*math.sqrt(math.pi)))
wp_gauss = normalisation * np.exp(1j * k * x_array - ((x_array - xc) ** 2) / (2 * (sigma ** 2)))

"wave packet réel & imaginaire"
#wave packet Real part 
wp_re=np.zeros(nx)
wp_re[:]=np.real(wp_gauss[:])
#wave packet Imaginary part 
wp_im=np.zeros(nx)
wp_im[:]=np.imag(wp_gauss[:])


#density = np.zeros((nt,nx))
#density[0,:] = np.absolute(wp_gauss[:]) ** 2

#final_density =np.zeros((n_frames,nx))

#Algo devant retourner la densité de probabilité de présence de la particule à différents instants
final_density, transmission = proba_particule(wp_gauss, V_potential, dx, dt, nt)
print(f"Taux de transmission : {transmission}")


"figure 1 pour la propagation"
plot_title = "E/Vo="+str(e)
energies, _ = etats_stationnaires(V_potential, dx)

fig = plt.figure() # initialise la figure principale
line, = plt.plot([], [])
plt.ylim(-1,1)
plt.xlim(0,2)
plt.plot(x_array,V_potential,label="Potentiel")
for E in energies:
    plt.axhline(E, color='orange', linestyle='--', linewidth=0.8, label=f"E = {E:.2f}")

plt.title(plot_title)
plt.xlabel("x")
plt.ylabel("Densité de probabilité de présence")
plt.legend() #Permet de faire apparaitre la legende

ani = animation.FuncAnimation(fig,animate,init_func=init, frames=n_frames, blit=False, interval=100, repeat=False)
#file_name = 'paquet_onde_e='+str(e)+'.mp4'
#ani.save(file_name, writer = animation.FFMpegWriter(fps=120, bitrate=5000))
plt.show()

"2e figures pour les états stationnaires"
energies, etats = etats_stationnaires(V_potential, dx)
print("Shape des états :", etats.shape) #verification & test
print("Énergies trouvées :", energies[:10]) #permet de voir chaque états (test)

for i in range(len(energies)):
    if energies[i] < 0 and energies[i] > -2:  # Ajuste cette plage pour mieux voir les états
        etat_norm = etats[:, i] / np.sqrt(np.sum(etats[:, i]**2) * dx)
        plt.plot(x_array, etat_norm**2, label=f"État {i+1}, E={energies[i]:.2f}")
plt.plot(x_array, V_potential, color='black', linestyle='--', label='Potentiel')
plt.title("États stationnaires")
plt.xlabel("x")
plt.ylabel("Densité de probabilité")
plt.legend()
plt.grid()
plt.ylim() # test
plt.show()

"3e figure pour la transmission analytiqu"
#appel à la fonction résolution analytique de l'équation Schrodinger
E_vals = np.linspace(0.1, 15, 300)  # valeurs d'énergie pour la courbe
a = E - v0
T_analytique = [resolution_schrodinger(E, -v0, a) for E in E_vals]  # largeur = 1.0 - 0.8 = 0.2


plt.plot(E_vals, T_analytique, label="Transmission analytique", color="green")
plt.axvline(E, linestyle='--', color='red', label=f"E utilisé = {E}")
plt.title("Transmission analytique en fonction de l’énergie")
plt.xlabel("Énergie")
plt.ylabel("T(E)")
plt.grid()
plt.legend()
plt.show()


"figure 4 pour comparer les prédictions avec les mesures expérimentales"

# données expérimentales (test)
E_exp = [1, 7, 14, 18, 20, 25, 30] #energie E
T_exp = [0.1, 0.3, 0.8, 0.4, 0.2, 0.1, 0.05]  #transmission T(E)(ici un exemple stylisé d’un effet Ramsauer)


# Transmission numérique
E_num, T_num = effet_ramsauer(V_potential, dx, dt, nt, x_array, v0)

# Transmission analytique
T_th = [resolution_schrodinger(E, -v0, a) for E in E_num]


"""
Analytique : orange pointillé
Numérique : ligne en bleu
Expérimental : point rouge
"""

plt.plot(E_num, T_num, label="Transmission numérique", color="blue")
plt.plot(E_num, T_th, label="Transmission analytique", color="orange", linestyle="--")
plt.scatter(E_exp, T_exp, label="Données expérimentales", color="red", marker='o')

plt.xlabel("Énergie (E)")
plt.ylabel("Transmission T(E)")
plt.title("Comparaison des transmissions : numérique, analytique, expérimentale")
plt.legend()
plt.grid(True)
plt.show()


