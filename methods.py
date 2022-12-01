import math
import numpy as np

# Line Search Methods
# Algorithm 3.1.- Backtracking Line Search
# Nocedal & Wright (2006), Numerical Optimization
def backtracking(fun,grad,xk,pk,alpha,rho,c):
    alpha_k = alpha
    fk = fun(xk)
    gk = grad(xk)
    while fun(xk + alpha_k * pk) > fk + c*alpha_k*(np.matmul(gk.T, pk)):
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
            alpha_s = zoom(alpha_im1, alpha_i)
            break
        phi_grad_alpha_i = phi_grad(alpha_i)
        if np.abs(phi_grad_alpha_i) <= -c2*phi_grad(0):
            alpha_s = alpha_i
            break
        if phi_grad_alpha_i >= 0:
            alpha_s = zoom(alpha_i, alpha_im1)
            break
        alpha_i = (alpha_i + alpha_max) / 2
        i = i + 1
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

def steepest_descend(fun, grad, lda, x0, eps, N):
    x = x0
    k = 0
    gk = grad(x)
    lk = lda(gk)
    nk = np.linalg.norm(gk)
    while nk >= eps and k <= N:
        x = x - np.matmul(lk, gk)
        f = fun(x)
        gk = grad(x)
        lk = lda(gk)
        nk = np.linalg.norm(gk)
        k = k + 1