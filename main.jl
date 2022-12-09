# -------------------------------------------------------------
# condición inicial
θ= 10 * rand(Normal(), 2)
# -------------------------------------------------------------
# parámetros del optimizador
α = 0.95
η = 0.9
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
f_params = (Xt=Xt.x1, yt=yt, κ=κ)
println("Optimización por métodos de primer orden")
# First order methods
Θ_GD = GD(θ, grad_exp, f_params, nIter, α)
println("GD, Inicio: $(Θ_GD[1,:]), -> Fin: $(Θ_GD[end,:])")

Θ_SGD = SGD(θ, grad_exp, f_params, nIter, α, batch_size)
println("SGD, Inicio: $(Θ_SGD[1,:]), -> Fin: $(Θ_SGD[end,:])")

Θ_MGD = MGD(θ, grad_exp, f_params, nIter, α, η)
println("MGD, Inicio: $(Θ_MGD[1,:]), -> Fin: $(Θ_MGD[end,:])")

Θ_NAG = NAG(θ, grad_exp, f_params, nIter, α, η)
println("NAG, Inicio: $(Θ_NAG[1,:]), -> Fin: $(Θ_NAG[end,:])")

#=
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