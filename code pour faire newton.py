import numpy as np

def decomp_LU(A):
    pivot(A)
    n=A.shape[0]
    U=A.copy()
    L=np.zeros(n)
    for i in range(n-1):
        for k in range(i+1,n):
            L[k,i]=U[k,i]/U[i,i]
            U[k,i:]=U[k,i:]-L[k,i]*U[i,i:]
    return L,U






def pivot(B):
    n=B.shape[0]
    B=B.copy()
    for k in range(n - 1):
        k0=k
        for i in range(k, n):
            if abs(B[i, k]) > abs(B[k0, k]):
                k0=i
        if k0!=k:
            B[[k, k0]]=B[[k0, k]]
    return B



def descente(L, b):
    n=len(b)
    y=np.zeros(n)
    for i in range(n):
        somme=0
        for j in range(i):
            somme+=L[i, j]*y[j]
        y[i]=(b[i]-somme)/L[i, i]
    return y





def remontee(U, y):
    n=len(y)
    x=np.zeros(n)
    for i in range(n - 1, -1, -1):
        somme=0
        for j in range(i + 1, n):
            somme+=U[i, j]*x[j]
        x[i]=(y[i]-somme)/U[i, i]
    return x