# Definición de librerías
using LinearAlgebra
using Distributions
using Statistics
using MLJBase
using Random
using Plots

function GD(θ0, grad, f_params, nIter, α)    
    #=
    Descenso de gradiente

    Parámetros
    -----------
    θ0        :   condicion inicial
    grad      :   función que calcula el gradiente
    gd_params :   lista de parametros para el algoritmo de descenso,
                     nIter = gd_params[0] número de iteraciones
                     alpha = gd_params[1] tamaño de paso alpha

    f_params  :   lista de parametros para la funcion objetivo
                     kappa = f_params['kappa'] parametro de escala (rechazo de outliers)
                     X     = f_params['X'] Variable independiente
                     y     = f_params['y'] Variable dependiente

    Regresa
    -----------
    Θ     :   trayectoria de los parametros    
    =#
    θ = θ0
    Θ = Vector{Float64}()
    for iter in 1:nIter
        p = grad(θ, f_params...)
        θ = θ - α * p
        append!(Θ, θ)
    end
    return Θ
end

function SGD(θ, grad, f_params, nIter, α, batch_size)
    #=
    Descenso de gradiente estocástico
    =#

    Θ = []
    for iter in 1:nIter
        # Obtenemos la muestra de tamaño (batch size)
        idxs = sample(axes(f_params.Xt, 1), batch_size)
        # sample
        Xt_sample = f_params.Xt[idxs]
        y_sample = f_params.yt[idxs]
        # parametros de la funcion objetivo
        f_params_sample = (Xt=f_params.Xt, yt=f_params.yt, κ=f_params.κ)
        p = grad(θ, f_params_sample...)
        θ = θ - α * p
        append!(Θ, θ)
    end
    return Θ
end

#=
def MGD(theta=[], grad=None, gd_params={}, f_params={}):
    '''
    Descenso de gradiente con momento (inercia)

    Parámetros
    -----------
    theta     :   condicion inicial
    grad      :   funcion que calcula el gradiente
    gd_params :   lista de parametros para el algoritmo de descenso,
                      nIter = gd_params['nIter'] número de iteraciones
                      alpha = gd_params['alpha'] tamaño de paso alpha
                      eta   = gd_params['eta']  parametro de inercia (0,1]
    f_params  :   lista de parametros para la funcion objetivo,
                      kappa = f_params['kappa'] parametro de escala (rechazo de outliers)
                      X     = f_params['X'] Variable independiente
                      y     = f_params['y'] Variable dependiente

    Regresa
    -----------
    Theta     :   trayectoria de los parametros
                     Theta[-1] es el valor alcanzado en la ultima iteracion
    '''
    nIter = gd_params['nIter']
    alpha = gd_params['alpha']
    eta = gd_params['eta']
    p_old = np.zeros(theta.shape)
    Theta = []
    for t in range(nIter):
        g = grad(theta, f_params=f_params)
        p = g + eta * p_old
        theta = theta - alpha * p
        p_old = p
        Theta.append(theta)
    return np.array(Theta)

def NAG(theta=[], grad=None, gd_params={}, f_params={}):
    '''
    Descenso acelerado de Nesterov

    Parámetros
    -----------
    theta     :   condicion inicial
    grad      :   funcion que calcula el gradiente
    gd_params :   lista de parametros para el algoritmo de descenso,
                      nIter = gd_params['nIter'] número de iteraciones
                      alpha = gd_params['alpha'] tamaño de paso alpha
                      eta   = gd_params['eta']  parametro de inercia (0,1]
    f_params  :   lista de parametros para la funcion objetivo,
                      kappa = f_params['kappa'] parametro de escala (rechazo de outliers)
                      X     = f_params['X'] Variable independiente
                      y     = f_params['y'] Variable dependiente

    Regresa
    -----------
    Theta     :   trayectoria de los parametros
                     Theta[-1] es el valor alcanzado en la ultima iteracion
    '''
    nIter = gd_params['nIter']
    alpha = gd_params['alpha']
    eta = gd_params['eta']
    p = np.zeros(theta.shape)
    Theta = []

    for t in range(nIter):
        pre_theta = theta - 2.0 * alpha * p
        g = grad(pre_theta, f_params=f_params)
        p = g + eta * p
        theta = theta - alpha * p
        Theta.append(theta)
    return np.array(Theta)

def ADADELTA(theta=[], grad=None, gd_params={}, f_params={}):
    '''
    Descenso de Gradiente Adaptable (ADADELTA)

    Parámetros
    -----------
    theta     :   condicion inicial
    grad      :   funcion que calcula el gradiente
    gd_params :   lista de parametros para el algoritmo de descenso,
                      nIter    = gd_params['nIter'] número de iteraciones
                      alphaADA = gd_params['alphaADADELTA'] tamaño de paso alpha
                      eta      = gd_params['eta']  parametro adaptación del alpha
    f_params  :   lista de parametros para la funcion objetivo,
                      kappa = f_params['kappa'] parametro de escala (rechazo de outliers)
                      X     = f_params['X'] Variable independiente
                      y     = f_params['y'] Variable dependiente

    Regresa
    -----------
    Theta     :   trayectoria de los parametros
                     Theta[-1] es el valor alcanzado en la ultima iteracion
    '''
    epsilon = 1e-8
    nIter = gd_params['nIter']
    alpha = gd_params['alphaADADELTA']
    eta = gd_params['eta']
    G = np.zeros(theta.shape)
    g = np.zeros(theta.shape)
    Theta = []
    for t in range(nIter):
        g = grad(theta, f_params=f_params)
        G = eta * g ** 2 + (1 - eta) * G
        p = 1.0 / (np.sqrt(G) + epsilon) * g
        theta = theta - alpha * p
        Theta.append(theta)
    return np.array(Theta)

def ADAM(theta=[], grad=None, gd_params={}, f_params={}):
    '''
    Descenso de Gradiente Adaptable con Momentum(A DAM)

    Parámetros
    -----------
    theta     :   condicion inicial
    grad      :   funcion que calcula el gradiente
    gd_params :   lista de parametros para el algoritmo de descenso,
                      nIter    = gd_params['nIter'] número de iteraciones
                      alphaADA = gd_params['alphaADAM'] tamaño de paso alpha
                      eta1     = gd_params['eta1'] factor de momentum para la direccion
                                 de descenso (0,1)
                      eta2     = gd_params['eta2'] factor de momentum para la el
                                 tamaño de paso (0,1)
    f_params  :   lista de parametros para la funcion objetivo,
                      kappa = f_params['kappa'] parametro de escala (rechazo de outliers)
                      X     = f_params['X'] Variable independiente
                      y     = f_params['y'] Variable dependiente

    Regresa
    -----------
    Theta     :   trayectoria de los parametros
                     Theta[-1] es el valor alcanzado en la ultima iteracion
    '''
    epsilon = 1e-8
    nIter = gd_params['nIter']
    alpha = gd_params['alphaADAM']
    eta1 = gd_params['eta1']
    eta2 = gd_params['eta2']
    p = np.zeros(theta.shape)
    v = 0.0
    Theta = []
    eta1_t = eta1
    eta2_t = eta2
    for t in range(nIter):
        g = grad(theta, f_params=f_params)
        p = eta1 * p + (1.0 - eta1) * g
        v = eta2 * v + (1.0 - eta2) * (g ** 2)
        # p = p/(1.-eta1_t)
        # v = v/(1.-eta2_t)
        theta = theta - alpha * p / (np.sqrt(v) + epsilon)
        eta1_t *= eta1
        eta2_t *= eta2
        Theta.append(theta)
    return np.array(Theta)
=#

# -------------------------------------------------------------
function func_quadratic(θ, Xt, yt)
    #=
    Funcion de costo
        sum_i (theta@x[i]-y[i])**2
    Parámetros:
    -----------
    θ   : variables del modelo
    Xt  : variables observadas
    yt  : variables a predecir
    =#
    return sum((θ[1] * Xt + θ[2] -yt).^2)
end

function grad_quadratic(θ, Xt, yt)
    #=
    Gradiente de la funcion de costo
        sum_i (theta@x[i]-y[i])**2
    Parámetros:
    -----------
    θ   : variables del modelo
    Xt  : variables observadas
    yt  : variables a predecir
    =#
    err = θ[1] * Xt + θ[2] - y
    ∂1 = err
    ∂2 = Xt * ∂1
    ∇ = [∂1, ∂2]
    return sum(∇)
end

# -------------------------------------------------------------
function func_exp(θ, Xt, yt, κ)
    #=
    Funcion de costo
        sum_i 1-exp(-κ(θ@x[i]-y[i])^2)
    =#
    err = Xt*θ[1] .+ θ[2] - yt
    return sum(1 .- exp.(-κ*err.^2))
end

function grad_exp(θ, Xt, yt, κ)
    #=
    Gradiente de la funcion de costo
        sum_i 1-exp(-κ(θ@x[i]-y[i])^2)
    =#
    err = Xt*θ[1] .+ θ[2] - yt
    ∂1 = err .* exp.(-κ*err.^2)
    ∂2 = Xt .* ∂1
    gradient = [mean(∂1), mean(∂2)]
    return gradient
end
# -------------------------------------------------------------
# condición inicial
θ= 10 * rand(Normal(), 2)
# -------------------------------------------------------------
# parámetros del optimizador
α = 0.95
batch_size = 100
# -------------------------------------------------------------
# parámetros de la función objetivo
κ = 0.01
# -------------------------------------------------------------
# máximo número de iteraciones
nIter = 300
# -------------------------------------------------------------
# datos iniciales
n_samples = 500
Xt, yt = make_regression(n_samples, 1; noise=0.5, rng=10)
# -------------------------------------------------------------
# verificación de función y gradiente
println("Verificación de vector inicial y evaluación de función y gradiente")
f = func_exp(θ, Xt.x1, yt, κ)
g = grad_exp(θ, Xt.x1, yt, κ)
println("θ : $(size(θ)), $(typeof(θ))")
println("fx: $(size(f)), $(typeof(f))")
println("∇ : $(size(g)), $(typeof(g))")

#= parámetros del algoritmo
             'alphaADADELTA'  : 0.7,
             'alphaADAM'      : 0.95,
             'mem_size'       : 20,
             'delta_0'        : 1,
             'gamma_0'        : 1,
             'eps'            : 1e-5,
             'eta'            : 0.9,
             'eta1'           : 0.9,
             'eta2'           : 0.999}
=#
println("Optimización por métodos de primer orden")
# First order methods
ΘGD = GD(θ, grad_exp, (Xt=Xt.x1, yt=yt, κ=κ), nIter, α)
println("GD, Inicio: $(ΘGD[1,:]), -> Fin: $(ΘGD[end,:])")

ThetaSGD = SGD(θ, grad_exp, (Xt=Xt.x1, yt=yt, κ=κ), nIter, α, batch_size)
println("SGD, Inicio: $(ΘGD[1,:]), -> Fin: $(ΘGD[end,:])")

#=
ThetaMGD = MGD(theta=theta, grad=grad_exp, gd_params=gd_params, f_params=f_params)
print('MGD, Inicio:', theta,'-> Fin:', ThetaMGD[-1,:])

ThetaNAG = NAG(theta=theta, grad=grad_exp, gd_params=gd_params, f_params=f_params)
print('NAG, Inicio:', theta,'-> Fin:', ThetaMGD[-1,:])

ThetaADADELTA = ADADELTA(theta=theta, grad=grad_exp, gd_params=gd_params, f_params=f_params)
print('ADADELTA, Inicio:', theta,'-> Fin:', ThetaADADELTA[-1,:])

ThetaADAM = ADAM(theta=theta, grad=grad_exp, gd_params=gd_params, f_params=f_params)
print('ADAM, Inicio:', theta,'-> Fin:', ThetaADAM[-1,:])
=#
println("Done!")

#=
Tmax=100
plt.figure(figsize=(10,10))

plt.subplot(211)
plt.plot(ThetaNAG[:Tmax], '.')
plt.title('NAG')

plt.subplot(212)
plt.plot(ThetaADAM[:Tmax], '.')
plt.title('ADAM')

plt.show()


import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt

mpl.rcParams['legend.fontsize'] = 14

fig = plt.figure(figsize=(15,15))
ax = fig.gca(projection='3d')
nIter=np.expand_dims(np.arange(ThetaGD.shape[0]),1)
Tmax=200
ax.plot(ThetaGD[:Tmax,0],  ThetaGD [:Tmax,1], nIter[:Tmax,0], label='GD')
ax.plot(ThetaSGD[:Tmax,0], ThetaSGD[:Tmax,1], nIter[:Tmax,0], label='SGD')
ax.plot(ThetaMGD[:Tmax,0], ThetaMGD[:Tmax,1], nIter[:Tmax,0], label='MGD')
ax.plot(ThetaNAG[:Tmax,0], ThetaNAG[:Tmax,1], nIter[:Tmax,0], label='NAG')
ax.plot(ThetaADADELTA[:Tmax,0], ThetaADADELTA[:Tmax,1], nIter[:Tmax,0], label='ADADELTA')
ax.plot(ThetaADAM[:Tmax,0], ThetaADAM[:Tmax,1], nIter[:Tmax,0], label='ADAM')
ax.legend()
ax.set_title(r'Trayectorias los parámetros calculados con distintos algoritmos')
ax.set_xlabel(r'$\theta_1$')
ax.set_ylabel(r'$\theta_0$')
ax.set_zlabel('Iteración')
plt.show()
=#