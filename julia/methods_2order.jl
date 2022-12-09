# Implementación de métodos de segundo orden
using LinearAlgebra
using Statistics
using MLJBase
include("TSsubproblem_solver_OBS.jl")

function LSR1TR(θ, func, grad, f_params, nIter, α, batch_size)
    S           = [];
    Y           = [];
    Ψ           = [];
    Minv        = [];

    tol         = 1e-5;
    delta       = 1;
    γ           = 1;

    epoch       = 10;
    lim_m       = 20;
    os          = 50;

    skip        = 0;
    k           = 0;
    epoch_k     = 0;
    Θ = θ
    
    for iter in 1:nIter
        #= println("=======================> Iteración $(iter), k=$(k) <==================="); =#
        f           = func(θ, f_params...)
        g           = grad(θ, f_params...)
        llgll       = norm(g)

        # Trust-Region subproblem
        if k == 0 || size(S,2) == 0
            p       = -delta*(g/llgll)
            Bp      =  γ*p
        else
            p       = TRsubproblem_solver_OBS(delta, γ, g, Ψ, Minv)
            Bp      = γ*p + Ψ*(Minv\(Ψ'*p))
        end

        Q_p         = (p'*(g + 0.5*Bp))[]
        llpll       = norm(p)

        # Actualización del vector de pesos
        θ_new       = θ + p
        f_new       = func(θ_new, f_params...)
        g_new       = grad(θ_new, f_params...)

        # Vectores de curvatura (s,y)
        s           = p
        y           = g_new - g
        
        # Reduction ratio
        ρ           = (f_new - f) / Q_p

        # Condición de aceptación del incremento
        θ = θ_new
        f = f_new
        g = g_new
        Θ = hcat(Θ, θ)

        # Condición de paro
        if llgll < tol
            println("Condición de paro")
            return
        end

        # Actualización del radio de la región de confianza
        if ρ > 0.75
            if norm(p) ≤ 0.8*delta
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
        if abs((s'*y_Bs)[]) > 1e-8*llpll*norm(y_Bs)
            #println("Curvature pairs")
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

            if size(S,2) == 0
                println("S is empty!")
            end

            while (size(S, 2) > 0)                
                SY      = S'*Y
                SS      = S'*S
                if size(SY) == ()
                    LDLt = SY
                else
                    LDLt = tril(SY) + tril(SY,-1)'
                end
                eig_val = eigen(LDLt,SS).values
                λHat_min= minimum(eig_val)
                if λHat_min > 0
                    γ   = max(0.5*λHat_min, 1e-6)
                else
                    γ   = min(1.5*λHat_min, -1e-6)
                end

                Minv    = (LDLt - γ*SS)
                Ψ       = Y - γ*S

                if size(Ψ,2) == rank(Ψ) && rank(Minv) == size(Minv,2)
                    break
                else
                    #println("Psi is NOT full column rank! or M is not invertible!")
                    S = S[:, 2:end]
                    Y = Y[:, 2:end]
                end
            end
        end

        k = k + 1;
    end
    return Θ
end


"""
    LSSR1TR(θ, func, grad, f_params, nIter, α, batch_size)

    La implementación del método SR1 con lote estocástico es básicamente una modificación del método
    anterior SR1 con región de confianza pero tomando ahora lotes de manera aleatoria durante el paso
    de la optimización y utilizar este lote para el cálculo del gradiente mas un factor de ajuste de 
    corrección del mismo
"""
function LSSR1TR(θ, func, grad, f_params, nIter, α, μ, batch_size)
    S           = [];
    Y           = [];
    Ψ           = [];
    Minv        = [];

    tol         = 1e-5;
    delta       = 1;
    γ           = 1;

    epoch       = 10;
    lim_m       = 20;
    os          = 50;

    skip        = 0;
    k           = 0;
    v_old       = 0;
    epoch_k     = 0;
    Θ           = θ;
    θ_old       = θ;
    
    for iter in 1:nIter
        #= println("=======================> Iteración $(iter), k=$(k) <==================="); =#
        # Al inicio de la iteración determinamos y obtenemos la muestra de tamaño (batch size)
        # sobre el conjunto de entrenamiento, este lote se utilizará para el cálculo del gradiente
        idxs = sample(axes(f_params.Xt, 1), batch_size)
        # sample
        Xt_sample = f_params.Xt[idxs]
        yt_sample = f_params.yt[idxs]
        # parametros de la funcion objetivo
        f_params_sample = (Xt=Xt_sample, yt=yt_sample, κ=f_params.κ)
        
        f           = func(θ, f_params_sample...)
        g           = grad(θ, f_params_sample...)
        llgll       = norm(g)

        # Trust-Region subproblem
        if k == 0 || size(S,2) == 0
            p       = -delta*(g/llgll)
            v_old   =  p
            Bp      =  γ*p
        else
            p       = TRsubproblem_solver_OBS(delta, γ, g, Ψ, Minv)
            # Calculamos factor de ajuste sobre el vector de dirección usando el valor de θ en la iteración previa
            v       = μ * v_old + (θ - θ_old)
            v       = μ * min(1.0, delta/norm(v)) * v
            v_old   = v
            # Actualizamos gradiente direccional
            p       = min(1.0, delta/norm(p + v)) * (p + v)
            Bp      = γ*p + Ψ*(Minv\(Ψ'*p))
        end        

        Q_p         = (p'*(g + 0.5*Bp))[]
        llpll       = norm(p)

        # Actualización del vector de pesos
        θ_new       = θ + p
        f_new       = func(θ_new, f_params_sample...)
        g_new       = grad(θ_new, f_params_sample...)

        # Vectores de curvatura (s,y)
        s           = p
        y           = g_new - g
        
        # Reduction ratio
        ρ           = (f_new - f) / Q_p

        # Condición de aceptación del incremento
        θ_old = θ # Para cálculo de factor de ajuste sobre el gradiente direccional
        θ = θ_new
        f = f_new
        g = g_new
        Θ = hcat(Θ, θ)
        # Condición de paro
        if llgll < tol
            println("Condición de paro")
            return
        end

        # Actualización del radio de la región de confianza
        if ρ > 0.75
            if norm(p) ≤ 0.8*delta
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
        if abs((s'*y_Bs)[]) > 1e-8*llpll*norm(y_Bs)
            #println("Curvature pairs")
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

            if size(S,2) == 0
                println("S is empty!")
            end

            while (size(S, 2) > 0)                
                SY      = S'*Y
                SS      = S'*S
                if size(SY) == ()
                    LDLt = SY
                else
                    LDLt = tril(SY) + tril(SY,-1)'
                end
                eig_val = eigen(LDLt,SS).values
                λHat_min= minimum(eig_val)
                if λHat_min > 0
                    γ   = max(0.5*λHat_min, 1e-6)
                else
                    γ   = min(1.5*λHat_min, -1e-6)
                end

                Minv    = (LDLt - γ*SS)
                Ψ       = Y - γ*S

                if size(Ψ,2) == rank(Ψ) && rank(Minv) == size(Minv,2)
                    break
                else
                    #println("Psi is NOT full column rank! or M is not invertible!")
                    S = S[:, 2:end]
                    Y = Y[:, 2:end]
                end
            end
        end

        k = k + 1;
    end
    return Θ
end