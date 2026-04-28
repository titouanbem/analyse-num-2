import numpy as np
import matplotlib.pyplot as plt


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


def inverse_LU(A):
    n = A.shape[0]
    A_inv = np.zeros((n, n))

    L, U = decomp_LU(A)

    for i in range(n):
        e = np.zeros(n)
        e[i] = 1

        y = descente(L, e)
        x = remontee(U, y)

        A_inv[:, i] = x

    return A_inv


def f(t, u):
    V, N, D, W, B = u

    V = max(V, 1e-8)
    N = max(N, 1e-8)
    D = max(D, 1e-8)
    W = max(W, 1e-8)
    B = max(B, 1e-8)

    dV = -0.1 * V + 0.05 * W
    dN =  0.1 * V - 0.02 * N
    dD =  0.03 * N - 0.01 * D
    dW = -0.04 * W + 0.02 * B
    dB =  0.05 * D - 0.03 * B

    return np.array([dV, dN, dD, dW, dB])


def jacobienne_numerique(G, x, eps=1e-6):
    n = len(x)
    J = np.zeros((n, n))
    Gx = G(x)

    for i in range(n):
        x_eps = x.copy()
        x_eps[i] += eps
        J[:, i] = (G(x_eps) - Gx) / eps

    return J


def newton(G, x0, tol=1e-8, max_iter=50):
    x = x0.copy()

    for k in range(max_iter):
        gx = G(x)
        J = jacobienne_numerique(G, x)

        J_inv = inverse_LU(J)

        x_new = x - J_inv @ gx

        if np.linalg.norm(x_new - x) < tol:
            return x_new

        x = x_new

    print("Attention : Newton n'a pas convergé")
    return x


def euler_implicite(f, u0, t0, tf, h):
    t = np.arange(t0, tf + h, h)
    u = np.zeros((len(t), len(u0)))

    u[0] = u0

    for n in range(len(t) - 1):
        un = u[n]
        tn1 = t[n + 1]

        def G(x):
            return x - un - h * f(tn1, x)

        u[n + 1] = newton(G, un)

    return t, u


u0 = np.array([10, 5, 2, 8, 1], dtype=float)

t0 = 0
tf = 100
h = 0.1

t, u = euler_implicite(f, u0, t0, tf, h)

V = u[:, 0]
N = u[:, 1]
D = u[:, 2]
W = u[:, 3]
B = u[:, 4]

plt.plot(t, V, label="V")
plt.plot(t, N, label="N")
plt.plot(t, D, label="D")
plt.plot(t, W, label="W")
plt.plot(t, B, label="B")

plt.xlabel("Temps")
plt.ylabel("Valeurs")
plt.title("Euler implicite avec Newton et inverse LU maison")
plt.legend()
plt.grid()
plt.show()