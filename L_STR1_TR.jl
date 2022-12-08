# Definición de librerías
using LinearAlgebra
using Statistics
using MLJBase
# Definición de la función
function func(w, X, y)
    κ = -0.01
    err = k.*(w[1] * X.x1 .+ w[2] - y)
    return sum(1 .- exp.(err.^2))
end
# Definición del gradiente
function grad(w, X, y)
    κ = 0.01
    err = w[1]*X.x1 .+ w[2] - y
    partial0 = err .* exp.(-κ.*err.^2)
    partial1 = X.x1 .* partial0
    return mean([partial1, partial0], dims=1)
end

S           = [];
Y           = [];

tol         = 1e-5;
delta           = 1;
γ           = 1;

epoch       = 10;
lim_m       = 20;
os          = 50;

skip        = 0;
k           = 0;
epoch_k     = 0;
# Definición de los datos de entrada (generados aleatoriamente)
pX, py      = make_regression(100, 1; noise=0.5, sparse=0.2, outliers=0.1)
# Definición del vector de entrada (generado aleatoriamente)
w           = [1,1]
while true
    println("=======================> Iteración $(k) <===================");
    f           = func(w, pX, py)
    g           = grad(w, pX, py)
    llgll       = norm(g)

    # Trust-Region subproblem
    if k == 0 || size(S,2) == 0
        p       = -delta*(g/llgll)
        Bp      =  γ*p
    else
        p       = TRsubproblem_solver_OBS(delta, γ, g, Ψ, Minv)
        Bp      = γ*p + Ψ*(Minv\(Ψ'*p))
    end

    Q_p         = p'*(g + 0.5*Bp)
    llpll       = norm(p)

    # Actualización del vector de pesos
    w_new       = w + p
    f_new       = func(w_new, pX, py)
    g_new       = grad(w_new, pX, py)

    # Vectores de curvatura (s,y)
    s           = p
    y           = g_new - g

    # Reduction ratio
    ρ           = (f_new - f) / Q_p

    # Condición de aceptación del incremento
    if ρ > 1e-4
    end

    # Condición de paro
    if llgll < tol
        println("Condición de paro")
        return
    end

    # Actualización del radio de la región de confianza
    if ρ > 0.75
        if norm(p) ≤ 0.8*Δ
            delta_new = delta
        else
            delta_new = 2*delta
        end
    elseif (0.1 ≤ ρ && ρ ≤ 0.75)
        delta_new = delta
    else
        delta_new = 0.5*delta
    end
    delta = delta_new

    # Condición de Actualización
    y_Bs        = y - Bp
    if abs(s'*y_Bs) > 1e-8*llpll*norm(y_Bs)
        if k == 0
            S = s
            Y = y
        else
            S = [S s]
            Y = [Y y]
        end
        if (size(S,2) > lim_m)
            S = S[:, 2:end]
            Y = Y[:, 2:end]
        end

        while (size(S, 2) > 0)
            println("Curvature pairs")
            SY      = S'*Y
            SS      = S'*S
            LDLt    = tril(SY) + tril(SY,-1)'
            eig_val = eigen(LDLt,SS).values
            λHat_min= minimum(eig_val)
            if λHat_min > 0
                γ   = max(0.5*λHat_min, 1e-6)
            else
                γ   = min(1.5*λHat_min, -1e-6)
            end

            Minv    = (LDLt - γ*SS)
            Psi     = Y - gamma*S

            if size(Psi,2) == rank(Psi) && rank(Minv) == size(Minv,2)
                break
            else
                println("Psi is NOT full column rank! or M is not invertible!")
                S = S[:, 2:end]
                Y = Y[:, 2:end]
            end
        end
    end

    global k = k + 1;
end