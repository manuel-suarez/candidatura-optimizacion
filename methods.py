import math
import numpy as np
import scipy.optimize as op

# Line Search Methods
# Algorithm 3.1.- Backtracking Line Search
# Nocedal & Wright (2006), Numerical Optimization
def backtracking(fun,grad,xk,pk):
    alpha_k = 1
    rho = 0.5
    c1 = 10e-4
    fk = fun(xk)
    gk = grad(xk)
    # Condición de Armijo
    while fun(xk + alpha_k * pk) > fk + c1*alpha_k*(gk.T @ pk):
        alpha_k = rho * alpha_k
    return alpha_k

# Algorithm 3.5.- Line Search Algorithm
# Nocedal & Wright (2006, Numerical Optimization, pp. 60
def line_search(fun,grad,xk,pk,alpha_max,c1,c2):
    def phi(alpha):
        return fun(xk + alpha*pk)
    def phi_grad(alpha):
        return pk.T @ grad(xk + alpha*pk)
    alpha_0 = 0
    alpha_1 = (alpha_0 + alpha_max) / 2
    alpha_im1 = alpha_1
    i = 1
    while True:
        phi_alpha_i = phi(alpha_i)
        if (phi_alpha_i > phi(0) + c1*alpha_i*phi_grad(0)) or (i > 1 and phi_alpha_i >= phi(alpha_im1)):
            alpha_s = zoom(phi, phi_grad, alpha_im1, alpha_i, c1, c2)
            break
        phi_grad_alpha_i = phi_grad(alpha_i)
        if np.abs(phi_grad_alpha_i) <= -c2*phi_grad(0):
            alpha_s = alpha_i
            break
        if phi_grad_alpha_i >= 0:
            alpha_s = zoom(phi, phi_grad, alpha_i, alpha_im1, c1, c2)
            break
        alpha_i = (alpha_i + alpha_max) / 2
        i = i + 1
    return alpha_s

# Algorithm 3.6.- Zoom
# Nocedal & Wright (2006, Numerical Optimization, pp. 61
def zoom(phi, phi_grad, alpha_lo, alpha_hi, c1, c2):
    alpha_s = 0
    while True:
        # Interpolación
        alpha_j = (alpha_lo + alpha_hi) / 2
        phi_alpha_j = phi(alpha_j)
        if (phi_alpha_j > phi(0) + c1*alpha_j*phi_grad(0)) or (phi_alpha_j >= phi(alpha_lo)):
            alpha_hi = alpha_j
        else:
            phi_grad_alpha_j = phi_grad(alpha_j)
            if np.abs(phi_grad_alpha_j) <= -c2*phi_grad(0):
                alpha_s = alpha_j
                break
            if phi_grad_alpha_j*(alpha_hi - alpha_lo) >= 0:
                alpha_hi = alpha_lo
            alpha_lo = alpha_j
    return alpha_s

def golden_search(start_interval, find_interval_size, eps, fun):
    rho = 0.381966011

    # Estimación de iteraciones
    N = math.log(eps/2)/(math.log((1 - rho)))
    # Intervalo inicial
    a0= start_interval
    b0= a0 + find_interval_size
    # Iteración inicial
    a1= a0 + rho * (b0 - a0)
    b1= a0 + (1 - rho) * (b0 - a0)
    # Evaluación de la función
    fa1 = fun(a1)
    fb1 = fun(b1)
    while True:
        if fa1 < fb1:
            b0= b1
            b1= a1
            fb1= fa1
            a1= a0 + rho * (b0 - a0)
            fa1= fun(a1)
        else:
            a0= a1
            a1= b1
            fa1= fb1
            b1= a0 + (1 - rho) * (b0 - a0)
            fb1= fun(b1)
        if (b0-a0) <= eps:
            break
    return a0, b0, N

# Nocedal & Wright (2006), Numerical Optimization
def steepest_descend(fun, grad, x0, eps, K):
    k = 0
    xk = x0
    gk = grad(xk)
    history = np.zeros((1, 2))  # storing x values
    history[0, :] = xk
    while np.linalg.norm(gk) >= eps and k <= K:
        # Dirección de descenso
        pk = -gk / np.linalg.norm(gk)
        # Line search with backtracking
        alphak = op.line_search(fun, grad, xk, pk)[0]
        print(xk.shape, pk.shape, alphak)
        # Actualizamos posición
        xk = xk + alphak * pk
        gk = grad(xk)
        # Guardamos en historial
        history = np.append(history, [xk], axis=0)  # storing x
        # Imprimimos iteración
        # print(f"Iteración: {k}, alpha: {alphak}")
        # Siguiente iteración
        k = k + 1
    return history

# Algorithm 6.1 BFGS Method
# Nocedal & Wright (2006), Numerical Optimization, pp. 140
def bfgs(fun, grad, x0, H0, eps, K):
    k = 0
    xk = 0
    I = np.eye(len(x0))
    Hk = H0.clone()
    gk = grad(xk)
    while np.linalg.norm(gk) > eps and k <= K:
        # Compute search direction
        pk = -Hk @ gk
        # Line search
        alphak = op.line_search(fun, grad, xk, pk)
        # Update xk
        xkp1 = xk + alphak * pk
        gkp1 = grad(xkp1)
        # Vectores de curvatura
        sk = xkp1 - xk
        yk = gkp1 - gk
        # Actualizamos Hk+1 con (6.17)
        rho_k = 1 / (yk.T @ sk)
        Hk = (I - rho_k*(sk@yk.T)) @ Hk @ (I - rho_k*(yk@sk.T)) + rho_k*(sk@sk.T)
        # Siguiente iteración
        xk = xkp1
        gk = gkp1
        k = k + 1
    return xk

# Algorithm 6.2 SR1 Trust-Region Method
# Nocedal & Wright (2006), Numerical Optimization, pp. 146
def SR1_TR(fun, grad, x0, B0, delta_0, eps, K):
    eta = 0.75
    r = 10e-8
    k = 0
    xk = x0
    fk = fun(xk)
    gk = grad(xk)
    delta_k = delta_0
    Bk = B0
    while np.linalg.norm(gk) > eps:
        # Solve the subproblem to find step and descend direction
        sk = solve()
        # Actualize difference of gradientes
        yk = grad(xk + sk)
        # Calculate reduction rate
        ared = fk - fun(xk + sk)
        pred = -(gk.T @ sk + 0.5*(sk.T @ Bk @ sk))
        reduc_rate = ared/pred
        # Increase/decrease trust region
        if reduc_rate > eta:
            xk1 = xk + sk
        else:
            xk1 = xk
        if reduc_rate > 0.75:
            if np.linalg.norm(sk) <= 0.85 * delta_k:
                delta_k1 = delta_k
            else:
                delta_k1 = 2 * delta_k
        else:
            if 0.1 <= reduc_rate and reduc_rate <= 0.75:
                delta_k1 = delta_k
            else:
                delta_k1 = 0.5 * delta_k
        # Actualize estimate of Hessian accord to SR1 update formula
        tmp = (yk - (Bk @ sk))
        if np.abs(sk.T @ tmp) >= r*np.linalg.norm(sk)*np.linalg.norm(tmp):
            Bk = Bk + (tmp @ tmp.T) / (tmp.T @ sk)
        k = k + 1

# Algorithm 1: L-SR1 Trust-Region (L-SR1-TR)
# Erway, et. al. (2019)
# TRUST-REGION ALGORITHMS FOR TRAINING RESPONSES:
# MACHINE LEARNING METHODS USING INDEFINITE
# HESSIAN APPROXIMATIONS
# pp. 8
def L_SR1_TR(fun, grad, x0, delta0, eps, gm0, tau1, tau2, tau3, eta1, eta2, eta3, eta4, alpha = 1, N = 10000):
    # 1.- Compute g0
    g0 = grad(x0)
    xk = x0
    gk = g0
    gmk = gm0
    deltak = delta0
    # 2.- For k = 0,1,2,... do
    for k in range(N):
        # 3-5.- If ||gk|| < eps return
        if np.linalg.norm(gk) <= eps:
            # Terminamos el ciclo
            break
        # 6.- Choose at most m pairs {sj, yj}

        # 7.- Compute p* using Algorithm 2
        pstar = Orthonormal_Basis_SR1()
        # 8.- Compute step-size alpha with Wolfe line-search on p*. Set p* = alpha*p*
        alpha = Wolfe_Linesearch()
        pstar = alpha*pstar
        # 9.- Compute the ratio rhok = (f(wk+p*)-f(wk))/Qk(p*)
        rhok = (fun(xk + pstar) - fun(xk)) / Q(pstar) # TODO Analizar qué es la expresión Q en el algoritmo
        # 10.- wk+1 = wk + p*
        xk1 = xk + pstar
        # 11.- Compute gk+1, sk, yk and gammak
        gk1 = grad(xk1)
        sk = xk1 - xk
        yk = gk1 - gk
        gmk = gm0 # TODO Analizar cómo se actualiza gamma_k
        # Ajuste de la región de confianza
        # 12.-
        if rhok < tau2:
            # 13.-
            deltak = np.min(eta1*deltak, eta2*np.linalg.norm(sk))
        else:
            # 15.-
            if rhok >= tau3 and np.linalg.norm(sk) >= eta3*deltak:
                # 16.-
                deltak = eta4*deltak
            else:
                # 18.-
                pass # deltak = deltak (no se actualiza)
        # Actualizamos cálculos
        xk = xk1
        fk = fun(xk)
        gk = grad(xk)
    # Finalizamos con retorno del valor mínimo encontrado
    return xk, fk, gk

# Algorithm 2: Orthonormal Basis SR1 Method
def Orthonormal_Basis_SR1(Yk, B0, Sk):
    Psik = Yk - B0 @ Sk
    # 1.- Compute the Cholesky factor R of Psi' * Psi
    R = np.linalg.cholesky(Psik.T @ Psik)
    # Requerimos construir la matriz M para el paso siguiente
    M = ()
    # 2.- Compute the spectral decomposition R*M*R' = U*Â*U' (solo usamos los eingenvalores)
    A = np.linalg.eig(R @ (M @ R.T))

def Wolfe_Linesearch():
    pass