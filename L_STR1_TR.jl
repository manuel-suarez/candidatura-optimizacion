# Definición de librerías
using LinearAlgebra
# Definición de la función
function func(w, X, y)
    
end
# Definición del gradiente
function grad(w, X, y)
    
end

S           = [];
Y           = [];

tol         = 1e-5;
δ           = 1;
γ           = 1;

epoch       = 10;
lim_m       = 20;
os          = 50;

skip        = 0;
k           = 0;
epoch_k     = 0;
# Definición de los datos de entrada (generados aleatoriamente)
X           = 0
y           = 0
# Definición del vector de entrada (generado aleatoriamente)
w           = [1,1]
while true
    println("=======================> Iteración $(k) <===================");
    f           = func(w, X, y)
    g           = grad(w, X, y)
    llgll       = norm(g)

    # Trust-Region subproblem
    if k == 0 || size(S,2) == 0
        p       = -δ*(g/llgll)
        Bp      =  γ*p
    else
        p       = TRsubproblem_solver_OBS(δ, γ, g, Ψ, Minv)
        Bp      = γ*p + Ψ*(Minv\(Ψ'*p))
    end

    Q_p         = p'*(g + 0.5*Bp)
    llpll       = norm(p)

    # Actualización del vector de pesos
    w_new       = w + p
    f_new       = func(w_new, X, y)
    g_new       = grad(w_new, X, y)

    # Vectores de curvatura (s,y)
    s           = p
    y           = g_new - g

    # Reduction ratio
    ρ           = (f_new - f) / Q_p


    global k = k + 1;
end