# Definición de la función
function func(theta, X, y)
    
end
# Definición del gradiente
function grad(theta, X, y)
    
end

S           = [];
Y           = [];

tol         = 1e-5;
delta       = 1;
gamma       = 1;

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
theta       = [1,1]
while true
    println("=======================> Iteración $(k) <===================");
    f = func(theta, X, y)

    global k = k + 1;
end