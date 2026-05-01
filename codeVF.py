import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
import time


# 1. PARAMÈTRES DU MODÈLE

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
V1 = 70.0    # Demi-saturation kappa_N
theta_N = 4.0    # Exposant theta-logistique
c_WN = 7.5       # Coeff mise à mort loup-wapiti
beta = 0.5       # Paramètre dépendance ratio
c_BN = 0.5       # Coeff mise à mort ours-wapiti
h_BN = 3.0       # Demi-saturation ours-wapiti
delta_N = 0.05   # Mortalité de fond

# Ongulé secondaire - Cerf (D)
r_D = 0.50       # Taux de croissance
K_D = 4.0        # Capacité de charge max
V2 = 50.0 # Demi-saturation kappa_D
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


# SYSTÈME D'ÉQUATIONS 

def systeme(t, u):
    # Sécurisation des valeurs pour éviter les erreurs de domaine
    V, N, D, W, B = np.maximum(u, 1e-8) 

    # Fonctions de capacité de charge dépendantes de la végétation
    kappa_N = K_N * (V / (V1 + V))
    kappa_D = K_D * (V / (V2 + V))

    # Taux de mise à mort total par loup
    f_WN = (c_WN * N) / (W + beta * N)
    f_WD = (c_WD * D) / (h_WD + D)
    Phi_W = f_WN + f_WD

    # Prédations ours
    f_BN = (c_BN * N) / (h_BN + N)
    f_BD = (c_BD * D) / (h_BD + D)

    # F1 Végétation
    dV_dt = r_V * V * (1 - V/K_V)  - ((alpha_N * V * N) / (nu_N + V)) - ((alpha_D * V * D) / (nu_D + V)) - ((alpha_B * V * B) / (nu_B + V))

    # F2 Ongulés principaux (N)
    dN_dt = r_N * N * (1 - (N/kappa_N)**theta_N) - (f_WN * W) - (f_BN * B) - (delta_N * N)

    # F3 Ongulés secondaires (D)
    dD_dt = r_D * D * (1 - (D/kappa_D)**theta_D) - (f_WD * W) - (f_BD * B) - (delta_D * D)

    # F4 Loups (W)
    dW_dt = (epsilon_1 * np.log(Phi_W + eta) - epsilon_2) * W - (m_W * W) - (mu_W * np.sin(np.pi * t)**2 * W)

    # F5 Ours (B)
    dB_dt = (e_B * (f_BN + f_BD) + (e_V * alpha_B * V) / (nu_B + V) - m_B) * B

    return np.array([dV_dt, dN_dt, dD_dt, dW_dt, dB_dt])



# MÉTHODES DE RÉSOLUTION NUMÉRIQUE


def methode_euler_explicite(f, u0, temps, dt):
    t = np.arange(temps[0], temps[1] + dt, dt)
    n = len(t)
    u = np.zeros((n, len(u0)))
    u[0] = u0
    for i in range(1, n):
        u[i] = u[i-1] + dt * f(t[i-1], u[i-1])
    return t, u

def methode_euler_implicite(f, u0, temps, dt):
    t = np.arange(temps[0], temps[1] + dt, dt)
    n = len(t)
    u = np.zeros((n, len(u0)))
    u[0] = u0
    for i in range(1, n):
        t_next = t[i]
        u_prev = u[i-1]
        def implicit_equation(u_next):
            return u_next - u_prev - dt * f(t_next, u_next)
        u[i] = fsolve(implicit_equation, u_prev)
    return t, u

def methode_rk4(f, u0, temps, dt):
    t = np.arange(temps[0], temps[1] + dt, dt)
    n = len(t)
    u = np.zeros((n, len(u0)))
    u[0] = u0
    for i in range(1, n):
        t_curr = t[i-1]
        u_curr = u[i-1]
        k1 = dt * f(t_curr, u_curr)
        k2 = dt * f(t_curr + dt/2, u_curr + k1/2)
        k3 = dt * f(t_curr + dt/2, u_curr + k2/2)
        k4 = dt * f(t_curr + dt, u_curr + k3)
        u[i] = u_curr + (k1 + 2*k2 + 2*k3 + k4) / 6
    return t, u


# LU


def jacobi(f, t, u, dt, eps=1e-8):
    n = len(u)
    J_f = np.zeros((n, n))
    f0 = f(t, u)
    for i in range(n):
        u_eps = np.copy(u)
        u_eps[i] += eps
        f1 = f(t, u_eps)
        J_f[:, i] = (f1 - f0) / eps
    I = np.eye(n)
    J_G = I - dt * J_f
    return J_G

def pivot(B):
    n = B.shape[0]
    B = B.copy().astype(float)

    for k in range(n - 1):
        k0 = k
        for i in range(k, n):
            if abs(B[i, k]) > abs(B[k0, k]):
                k0 = i

        if k0 != k:
            B[[k, k0]] = B[[k0, k]]

    return B

def decomp_LU(A):
    A = pivot(A)
    n = A.shape[0]

    U = A.copy()
    L = np.eye(n)

    for i in range(n - 1):
        for k in range(i + 1, n):
            L[k, i] = U[k, i] / U[i, i]
            U[k, i:] = U[k, i:] - L[k, i] * U[i, i:]

    return L, U

def descente(L, b):
    n = len(b)
    y = np.zeros(n)

    for i in range(n):
        somme = 0
        for j in range(i):
            somme += L[i, j] * y[j]

        y[i] = (b[i] - somme) / L[i, i]

    return y

def remontee(U, y):
    n = len(y)
    x = np.zeros(n)

    for i in range(n - 1, -1, -1):
        somme = 0
        for j in range(i + 1, n):
            somme += U[i, j] * x[j]

        x[i] = (y[i] - somme) / U[i, i]

    return x

def methode_euler_implicite_lu(f, u0, temps, dt, tol=1e-6, max_iter=50):
    t = np.arange(temps[0], temps[1] + dt, dt)
    n = len(t)
    u = np.zeros((n, len(u0)))
    u[0] = u0
    for i in range(1, n):
        t_next = t[i]
        u_prev = u[i-1]
        u_x = np.copy(u_prev)
        
        for iteration in range(max_iter):
            f_val = f(t_next, u_x)
            G = u_x - u_prev - dt * f_val
            if np.linalg.norm(G) < tol:
                break
            J_G = jacobi(f, t_next, u_x, dt)
            L, U = decomp_LU(J_G)
            y = descente(L, -G)
            delta_u = remontee(U, y)
            u_x += delta_u
        u[i] = u_x
    return t, u



# 5. COMPARAISON DES TEMPS D'EXÉCUTION

def compare_methods_time():
    print("--- DÉMARRAGE DU BENCHMARK DES MÉTHODES ---")
    t_span_bench = (0, 50)  
    u0_bench = [440.0, 4.5, 2.5, 0.0, 0.25]
    dt_bench = 0.1
    
    # Euler Explicite
    start = time.time()
    methode_euler_explicite(systeme, u0_bench, t_span_bench, dt_bench)
    time_exp = time.time() - start
    print(f"Euler Explicite       : {time_exp:.5f} s")
    
    # RK4
    start = time.time()
    methode_rk4(systeme, u0_bench, t_span_bench, dt_bench)
    time_rk4 = time.time() - start
    print(f"Runge-Kutta 4 (RK4)   : {time_rk4:.5f} s")
    
    # Euler Implicite (LU)
    start = time.time()
    methode_euler_implicite_lu(systeme, u0_bench, t_span_bench, dt_bench)
    time_imp_lu = time.time() - start
    print(f"Euler Implicite (LU)  : {time_imp_lu:.5f} s")
    
    # Euler Implicite (SciPy fsolve)
    start = time.time()
    methode_euler_implicite(systeme, u0_bench, t_span_bench, dt_bench)
    time_imp_fsolve = time.time() - start
    print(f"Euler Imp. (fsolve)   : {time_imp_fsolve:.5f} s")
    print("-------------------------------------------\n")
    
    
    fig_time, ax_time = plt.subplots(figsize=(8, 5))
    methodes = ['Euler\nExplicite', 'RK4', 'Euler Imp.\n(LU Manuel)', 'Euler Imp.\n(fsolve)']
    temps = [time_exp, time_rk4, time_imp_lu, time_imp_fsolve]
    bars = ax_time.bar(methodes, temps, color=['#2ca02c', '#1f77b4', '#ff7f0e', '#d62728'])
    
    ax_time.set_ylabel('Temps d\'exécution (secondes)')
    ax_time.set_title("Comparaison des temps de calcul (Années 0 à 50)")
    
    
    for bar in bars:
        yval = bar.get_height()
        ax_time.text(bar.get_x() + bar.get_width()/2, yval + max(temps)*0.02, 
                     f"{yval:.4f} s", ha='center', va='bottom', fontweight='bold')
        
    plt.tight_layout()



# 6. SIMULATION ET AFFICHAGE : SCÉNARIO YELLOWSTONE

if __name__ == "__main__":
    
    compare_methods_time()

    
    
    
    
    dt = 0.1
    
   
    # PHASE 1 : L'écosystème sans les loups (Années 0 à 50)
  
    print("Calcul de la Phase 1 (Sans loups)...")
    t_span_phase1 = (0, 50)
    u0_phase1 = [440.0, 4.5, 2.5, 0.04, 0.25] 
    
    
    t_phase1, u_phase1 = methode_euler_implicite_lu(systeme, u0_phase1, t_span_phase1, dt)
    
    
    # PHASE 2 : Réintroduction des loups (Années 50 à 150)
    
    print("Calcul de la Phase 2 (Réintroduction en cours)...")
    t_span_phase2 = (50, 150)
    u0_phase2 = np.copy(u_phase1[-1])
    u0_phase2[3] = 0.04  
    t_span_phase2_decale = (50 + dt, 150)
    
    t_phase2, u_phase2 = methode_euler_implicite_lu(systeme, u0_phase2, t_span_phase2_decale, dt)
    
    
    
  
    print("Calculs terminés ! Génération des graphiques...")
    t_total = np.concatenate((t_phase1, t_phase2))
    u_total = np.concatenate((u_phase1, u_phase2))
    
    V, N, D, W, B = u_total.T

    
    fig, axs = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    annee_reintro = 50
    
    # Végétation
    axs[0].plot(t_total, V, color='green', label='Végétation (V)')
    axs[0].axvline(x=annee_reintro, color='gray', linestyle='--', label='Réintroduction')
    axs[0].set_ylabel('Biomasse (kg/km²)')
    axs[0].legend()
    axs[0].grid(True)
    axs[0].set_title("Dynamique de la Végétation")

    # Ongulés
    axs[1].plot(t_total, N, color='blue', label='Ongulés principaux (N)')
    axs[1].plot(t_total, D, color='cyan', label='Ongulés secondaires (D)')
    axs[1].axvline(x=annee_reintro, color='gray', linestyle='--')
    axs[1].set_ylabel('Densité (animaux/km²)')
    axs[1].legend()
    axs[1].grid(True)
    axs[1].set_title("Dynamique des Proies")

    #  Prédateurs
    axs[2].plot(t_total, W, color='red', label='Loups (W)')
    axs[2].plot(t_total, B, color='orange', label='Ours (B)')
    axs[2].axvline(x=annee_reintro, color='gray', linestyle='--')
    axs[2].set_ylabel('Densité (animaux/km²)')
    axs[2].set_xlabel('Temps (Années)')
    axs[2].legend()
    axs[2].grid(True)
    axs[2].set_title("Dynamique des Prédateurs")

    plt.tight_layout()
    plt.show()