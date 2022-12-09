# Implementación de métodos de segundo orden
using LinearAlgebra
using Statistics
using MLJBase

function LSR1TR(θ, func, grad, f_params, nIter, α, batch_size)
    S           = [];
    Y           = [];

    tol         = 1e-5;
    delta       = 1;
    γ           = 1;

    epoch       = 10;
    lim_m       = 20;
    os          = 50;

    skip        = 0;
    k           = 0;
    epoch_k     = 0;

    for iter in 1:nIter
        println("=======================> Iteración $(iter), k=$(k) <===================");
        f           = func(θ, f_params...)
        g           = grad(θ, f_params...)
        grad(θ, f_params...)
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
        println("s=$(s)")
        println("y=$(y)")

        # Reduction ratio
        println(size(f_new - f))
        println(size(Q_p))
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
            println("Curvature pairs")
            if k == 0
                S = s
                Y = y
            else
                S = [S s]
                Y = [Y y]
            end
            println(S)
            println(Y)
            if (size(S,2) > lim_m)
                S = S[:, 2:end]
                Y = Y[:, 2:end]
            end

            while (size(S, 2) > 0)                
                SY      = S'*Y
                SS      = S'*S
                println(SY)
                if size(SY) == ()
                    LDLt = SY
                else
                    LDLt = tril(SY) + tril(SY,-1)'
                end
                println(LDLt)
                eig_val = eigen(LDLt,SS).values
                λHat_min= minimum(eig_val)
                if λHat_min > 0
                    γ   = max(0.5*λHat_min, 1e-6)
                else
                    γ   = min(1.5*λHat_min, -1e-6)
                end

                Minv    = (LDLt - γ*SS)
                Psi     = Y - γ*S

                println(Psi)
                if size(Psi,2) == rank(Psi) && rank(Minv) == size(Minv,2)
                    break
                else
                    println("Psi is NOT full column rank! or M is not invertible!")
                    S = S[:, 2:end]
                    Y = Y[:, 2:end]
                end
            end
        end

        k = k + 1;
    end
end