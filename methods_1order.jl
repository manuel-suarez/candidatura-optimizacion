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

#= Descenso de gradiente estocástico =#
function SGD(θ, grad, f_params, nIter, α, batch_size)
    Θ = []
    for iter in 1:nIter
        # Obtenemos la muestra de tamaño (batch size)
        idxs = sample(axes(f_params.Xt, 1), batch_size)
        # sample
        Xt_sample = f_params.Xt[idxs]
        yt_sample = f_params.yt[idxs]
        # parametros de la funcion objetivo
        f_params_sample = (Xt=Xt_sample, yt=yt_sample, κ=f_params.κ)
        p = grad(θ, f_params_sample...)
        θ = θ - α * p
        append!(Θ, θ)
    end
    return Θ
end

#= Descenso de gradiente con momento (inercia) =#
function MGD(θ, grad, f_params, nIter, α, η)
    p_old = zeros(size(θ))
    Θ = []
    for iter in 1:nIter
        g = grad(θ, f_params...)
        p = g + η * p_old
        θ = θ - α * p
        p_old = p
        append!(Θ, θ)
    end
    return Θ
end

#= Descenso acelerado de Nesterov =#
function NAG(θ, grad, f_params, nIter, α, η)
    p = zeros(size(θ))
    Θ = []

    for iter in 1:nIter
        pre_θ = θ - 2.0 * α * p
        g = grad(pre_θ, f_params...)
        p = g + η * p
        θ = θ - α * p
        append!(Θ, θ)
    end
    return Θ
end

#=
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

