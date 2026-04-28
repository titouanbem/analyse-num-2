import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

# ==========================================
# 1. PARAMÈTRES DU MODÈLE
# ==========================================
# Végétation (V)
r_V = 1.20       # Taux de croissance intrinsèque
K_V = 500.0      # Capacité de charge
alpha_N = 15.0   # Intensité pâturage wapitis
nu_N = 200.0     # Demi-saturation wapitis
alpha_D = 8.0    # Intensité pâturage cerfs
nu_D = 100.0     # Demi-saturation cerfs
alpha_B = 3.0    # Intensité pâturage ours
nu_B = 120.0     # Demi-saturation ours

# Ongulé principal - Wapiti/Orignal (N)
r_N = 0.30       # Taux de croissance
K_N = 7.0        # Capacité de charge max
V_star = 70.0    # Demi-saturation kappa_N
theta_N = 4.0    # Exposant theta-logistique
c_WN = 7.5       # Coeff mise à mort loup-wapiti
beta = 0.5       # Paramètre dépendance ratio
c_BN = 0.5       # Coeff mise à mort ours-wapiti
h_BN = 3.0       # Demi-saturation ours-wapiti
delta_N = 0.05   # Mortalité de fond

# Ongulé secondaire - Cerf (D)
r_D = 0.50       # Taux de croissance
K_D = 4.0        # Capacité de charge max
V_star_star = 50.0 # Demi-saturation kappa_D
theta_D = 2.0    # Exposant theta-logistique
c_WD = 2.0       # Coeff mise à mort loup-cerf
h_WD = 2.0       # Demi-saturation loup-cerf
c_BD = 0.3       # Coeff mise à mort ours-cerf
h_BD = 1.5       # Demi-saturation ours-cerf
delta_D = 0.08   # Mortalité de fond

# Loups (W)
epsilon_1 = 0.25 # Pente log de la réponse numérique
epsilon_2 = 0.5  # Décalage de base
eta = 0.01       # Décalage régularisation log
m_W = 0.15       # Mortalité naturelle
mu_W = 0.10      # Amplitude prélèvement saisonnier humain

# Ours (B)
e_B = 0.12       # Gain énergétique proies
e_V = 0.02       # Gain énergétique végétation
m_B = 0.08       # Mortalité naturelle

# ==========================================
# 2. SYSTÈME D'ÉQUATIONS D'ÉTAT (F(t, u))
# ==========================================
def system_dynamics(t, u):
    # Sécurisation des valeurs pour éviter les erreurs de domaine (ex: log(negatif) ou division par zéro)
    V, N, D, W, B = np.maximum(u, 1e-8) 

    # Fonctions de capacité de charge dépendantes de la végétation
    kappa_N = K_N * (V / (V_star + V))
    kappa_D = K_D * (V / (V_star_star + V))

    # Taux de mise à mort total par loup
    f_WN = (c_WN * N) / (W + beta * N)
    f_WD = (c_WD * D) / (h_WD + D)
    Phi_W = f_WN + f_WD

    # Prédations ours
    f_BN = (c_BN * N) / (h_BN + N)
    f_BD = (c_BD * D) / (h_BD + D)

    # F1 : Végétation
    dV_dt = r_V * V * (1 - V/K_V) \
            - (alpha_N * V * N) / (nu_N + V) \
            - (alpha_D * V * D) / (nu_D + V) \
            - (alpha_B * V * B) / (nu_B + V)

    # F2 : Ongulés principaux (N)
    dN_dt = r_N * N * (1 - (N/kappa_N)**theta_N) \
            - f_WN * W \
            - f_BN * B \
            - delta_N * N

    # F3 : Ongulés secondaires (D)
    dD_dt = r_D * D * (1 - (D/kappa_D)**theta_D) \
            - f_WD * W \
            - f_BD * B \
            - delta_D * D

    # F4 : Loups (W) - non autonome avec prélèvement humain saisonnier
    dW_dt = (epsilon_1 * np.log(Phi_W + eta) - epsilon_2) * W \
            - m_W * W \
            - mu_W * np.sin(np.pi * t)**2 * W

    # F5 : Ours (B) - omnivores
    dB_dt = (e_B * (f_BN + f_BD) + (e_V * alpha_B * V) / (nu_B + V) - m_B) * B

    return np.array([dV_dt, dN_dt, dD_dt, dW_dt, dB_dt])


# ==========================================
# 3. MÉTHODES DE RÉSOLUTION NUMÉRIQUE
# ==========================================

def solve_euler_implicite(f, u0, t_span, dt):
    """
    Résolution par la méthode d'Euler Implicite.
    Nécessite la résolution de u_{n+1} - u_n - dt * f(t_{n+1}, u_{n+1}) = 0
    """
    t = np.arange(t_span[0], t_span[1] + dt, dt)
    n_steps = len(t)
    u = np.zeros((n_steps, len(u0)))
    u[0] = u0
    
    for i in range(1, n_steps):
        t_next = t[i]
        u_prev = u[i-1]
        
        # Fonction à annuler pour fsolve
        def implicit_equation(u_next):
            return u_next - u_prev - dt * system_dynamics(t_next, u_next)
        
        # Le guess initial est l'état précédent (u_prev)
        u[i] = fsolve(implicit_equation, u_prev)
        
    return t, u

def solve_rk4(f, u0, t_span, dt):
    """
    Résolution par la méthode de Runge-Kutta d'ordre 4.
    """
    t = np.arange(t_span[0], t_span[1] + dt, dt)
    n_steps = len(t)
    u = np.zeros((n_steps, len(u0)))
    u[0] = u0
    
    for i in range(1, n_steps):
        t_curr = t[i-1]
        u_curr = u[i-1]
        
        k1 = dt * system_dynamics(t_curr, u_curr)
        k2 = dt * system_dynamics(t_curr + dt/2, u_curr + k1/2)
        k3 = dt * system_dynamics(t_curr + dt/2, u_curr + k2/2)
        k4 = dt * system_dynamics(t_curr + dt, u_curr + k3)
        
        u[i] = u_curr + (k1 + 2*k2 + 2*k3 + k4) / 6
        
    return t, u

# ==========================================
# 4. SIMULATION ET AFFICHAGE
# ==========================================
if __name__ == "__main__":
    # Conditions initiales
    u0 = [440, 4.5, 2.5, 0.04, 0.25]
    t_span = (0, 100) # Simulation sur 100 ans pour bien voir la dynamique
    dt = 0.1          # Pas de temps (0.1 an)

    print("Calcul avec Euler Implicite...")
    t_ei, u_ei = solve_euler_implicite(system_dynamics, u0, t_span, dt)
    
    print("Calcul avec Runge-Kutta 4...")
    t_rk, u_rk = solve_rk4(system_dynamics, u0, t_span, dt)

    
    V, N, D, W, B = u_ei.T

    # Création des graphiques
    fig, axs = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    
    # Graphe 1 : Végétation
    axs[0].plot(t_ei, V, color='green', label='Végétation (V)')
    axs[0].set_ylabel('Biomasse (kg/km²)')
    axs[0].legend()
    axs[0].grid(True)
    axs[0].set_title("Dynamique de la Végétation")

    # Graphe 2 : Ongulés
    axs[1].plot(t_ei, N, color='blue', label='Ongulés principaux (N)')
    axs[1].plot(t_ei, D, color='cyan', label='Ongulés secondaires (D)')
    axs[1].set_ylabel('Densité (animaux/km²)')
    axs[1].legend()
    axs[1].grid(True)
    axs[1].set_title("Dynamique des Proies")

    # Graphe 3 : Prédateurs
    axs[2].plot(t_ei, W, color='red', label='Loups (W)')
    axs[2].plot(t_ei, B, color='orange', label='Ours (B)')
    axs[2].set_ylabel('Densité (animaux/km²)')
    axs[2].set_xlabel('Temps (Années)')
    axs[2].legend()
    axs[2].grid(True)
    axs[2].set_title("Dynamique des Prédateurs")

    plt.tight_layout()
    plt.show()