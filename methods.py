import math
import numpy as np

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
def line_search(phi,phi_grad,alpha_max,alpha_1,c1,c2):
    alpha_i = 0
    alpha_im1 = 0
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

def steepest_descend(fun, grad, x0, eps, K):
    k = 0
    xk = x0
    history = np.zeros((1, 2))  # storing x values
    history[0, :] = xk
    while np.linalg.norm(grad(xk)) >= eps and k <= K:
        # Dirección de descenso
        pk = -grad(xk)
        # Line search with backtracking
        alphak = backtracking(fun, grad, xk, pk)
        # Actualizamos posición
        xk = xk + alphak * pk
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
        # TODO reemplazar por line_search
        alphak = backtracking(fun, grad, xk, pk)
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