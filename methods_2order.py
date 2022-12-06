import numpy as np

# Definimos algunas variables para acortar los nombres de funciones
NORM = np.linalg.norm
CHOL = np.linalg.cholesky
EIG  = np.linalg.eig
INV  = np.linalg.inv

# Algorithm 1: L-SR1 Trust-Region (L-SR1-TR) Method
# Erway, et. al., p. 8
def L_STR1_TR(theta_0=[], func=None, grad=None, gd_params={}, f_params={}):
    '''
        Stochastic SR1 Trust-region scheme (MB-LSR1)
        Griffin, et. al.
        A minibatch stochastic Quasi-Newton method adapted for nonconvex deep learning problems
        Algorithm 1, p. 5

        Parámetros
        -----------
        theta     :   condicion inicial (x0)
        fun       :   función de costo
        grad      :   gradiente de la función de costo
        gd_params :   lista de parametros para el algoritmo de descenso,
                        nIter    = gd_params['nIter'] número de iteraciones
                        numIter:            Max. num. of iterations
                        batch_size:         Batch size
                        mmr:                Memory size
                        delta_0:            Trust-region radius
                        eps:                Termination tolerance
                        gamma_0:
                        tau_1:
                        tau_2:
                        tau_3:
                        eta_1:
                        eta_2:
                        eta_3:
                        eta_4
                        alpha:              Step-size for Wolfe line-search (alpha = 1)
                        mu:                 (mu = 0.9)
        f_params  :   lista de parametros para la funcion objetivo,
                          kappa = f_params['kappa'] parametro de escala (rechazo de outliers)
                          X     = f_params['X'] Variable independiente
                          y     = f_params['y'] Variable dependiente

        Regresa
        -----------
        Theta     :   trayectoria de los parametros
                         Theta[-1] es el valor alcanzado en la ultima iteracion
    '''
    nIter       = gd_params['nIter']
    numIter     = gd_params['numIter']
    batch_size  = gd_params['batch_size']
    mmr         = gd_params['mmr']
    delta_0	    = gd_params['delta_0']
    eps	        = gd_params['eps']
    gamma_0	    = gd_params['gamma_0']
    tau_1	    = gd_params['tau_1']
    tau_2	    = gd_params['tau_2']
    tau_3	    = gd_params['tau_3']
    eta_1	    = gd_params['eta_1']
    eta_2	    = gd_params['eta_2']
    eta_3	    = gd_params['eta_3']
    eta_4       = gd_params['eta_4']
    alpha	    = gd_params['alpha']
    mu	        = gd_params['mu']
    # Variables para el muestreo aleatorio de los lotes (batch size)
    (high, dim) = f_params['X'].shape
    # 1.- Compute initial batch I0 and g0
    smpIdx = np.random.randint(low=0, high=high, size=batch_size, dtype='int32')
    # sample
    smpX = f_params['X'][smpIdx]
    smpy = f_params['y'][smpIdx]
    # Inicialización de variables
    S = []
    Y = []
    Theta = []
    theta_k = theta_0
    delta_k = delta_0
    # g_k es el gradiente completo, gh_k es el gradiente de un lote seleccionado aleatoriamente
    initial_batch = {'kappa': f_params['kappa'], 'X': smpX, 'y': smpy}
    batch_k = initial_batch
    fh_k = func(theta_k, f_params=batch_k)
    gh_k = grad(theta_k, f_params=batch_k)                                      # Línea 1
    for iter in range(numIter):                                                 # Línea 2
        if NORM(gh_k) <= eps:                                                   # Línea 3
            break                                                               # Línea 4.- Terminamos el ciclo para retonar el valor de la función
        # endif                                                                 # Línea 5
        # Choose at most m pairs {sj, yj}
            # Line 6
        # Compute p* using Algorithm 2
            # Line 7
        # Compute step-size alpha with Wolfe line-search on p_star. Set p_star = alpha * p_star
            # Compute step-size alpha with Wolfe line-search on p*. Set p* = alpha p*
            alpha_k = alpha


            # Line 8
        # Line-search end
        # Compute the ratio rho_k = actual reduction / predicted reduction
        ared = func(theta_k + p_s) - func(theta_k)
        pred =
        rho_k = ared/pred                                                       # Línea 9
        theta_kp1 = theta_k + p_s                                               # Línea 10
        fh_kp1 = func(theta_kp1, f_params=batch_k)
        # Compute gh_k+1, sk, yk and gammak
        gh_kp1 = grad(theta_kp1, f_params=batch_k)                              # Línea 11 gh_k+1
        sk = fh_kp1 - fh_k                                                      # Línea 11 sk
        yk = gh_kp1 - gh_k                                                      # Línea 11 yk
        S.append(sk)
        Y.append(yk)
        norm_sk = NORM(s_k)
        if rho_k < tau_2:                                                       # Línea 12
            delta_kp1 = min(eta1 * delta_k, eta_2 * norm_sk)                    # Línea 13
        else:                                                                   # Línea 14
            if rho_k >= tau_3 and norm_sk >= eta_3 * delta_k:                   # Línea 15
                delta_kp1 = eta_4 * delta_k                                     # Línea 16
            else:                                                               # Línea 17
                delta_kp1 = delta_k                                             # Línea 18
            # endif                                                             # Línea 19
        # endif                                                                 # Línea 20
        # Actualizamos variables de iteración
        theta_k = theta_kp1
        delta_k = delta_kp1
        Theta.append(theta)
    # endfor                                                                    # Línea 21
    return np.array(Theta)

def TRsubproblem_solver_OBS:
    Psi = Y_k - B0 @ S_k
    M = INV(D_k + L_k + L_k.T - S_k.T @ (B0 @ S_k))
    # Compute the Cholesky factor R of PsiT @ Psi
    R = CHOL(Psi.T.dot(Psi))                            # Línea 1
    # Compute the spectral decomposicion R @ M @ Rt = U A_h Ut (with lamba_h1 <= ... <= lamba_hk)
    A, U = EIG(R.dot(M.dot(R.T)))

# Algorithm 1.- Stochastic SR1 Trust-region scheme (MB-LSR1)
# Griffin, et. al., p. 5
def MB_LSR1(theta=[], fun=None, grad=None, gd_params={}, f_params={}):
    '''
    Stochastic SR1 Trust-region scheme (MB-LSR1)
    Griffin, et. al.
    A minibatch stochastic Quasi-Newton method adapted for nonconvex deep learning problems
    Algorithm 1, p. 5

    Parámetros
    -----------
    theta     :   condicion inicial
    fun       :   función de costo
    grad      :   gradiente de la función de costo
    gd_params :   lista de parametros para el algoritmo de descenso,
                    nIter    = gd_params['nIter'] número de iteraciones
                    X,y:                Observations and labels of the data to classify
                    seed:               Random seeder
                    numIter:            Max. num. of iterations
                    mmr:                Memory size
                    r_mmr:              Restart memory size r_mmr <= m
                    n:                  Initial batch size
                    tau:                Restart tolerance
                    K:                  Progress check frequency
                    gamma_1, gamma2:    Progress threshold parameters
                    zeta:               Progressive radius parameter [0, 1]
                    mu:                 Momentum
                    alpha_s:            Learning rate [0, 1]
                    radius:             Radius of the sampling region
                    eps:                Epsilon criterion to sample S,Y pairs
                    eta:                Trust region increase/reduce criterion
                    delta_init:         Initial radius of trust region
                    epsTR:              Epsilon criterion for trust region subproblem
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
    # Line 1.- Input
    N       = gd_params['N']        # Num iterations
    mmr     = gd_params['mmr']      # Memory
    mmr_h   = gd_params['mmr_h']    # Restart memory
    n       = gd_params['n']        # Initial batch-size
    delta_0 = gd_params['delta_0']  # Initial radius (delta_0)
    # Line 2.- Parameters
    tau     = gd_params['tau']      # Restart tolerance
    K       = gd_params['K']        # Progress check frequency
    gamma_1 = gd_params['gamma_1']  # Progress threshold parameter 1
    gamma_2 = gd_params['gamma_2']  # Progress threshold parameter 2
    nh_k    = gd_params['nh_k']     # Progress check batch-size
    zeta    = gd_params['zeta']     # Progressive radius parameter [0, 1]
    eta     = gd_params['eta']      # Step parameter
    mu      = gd_params['mu']       # Momentum
    alpha_s = gd_params['alpha_s']  # Learning-rate [0, 1]
    # Parámetros adicionales
    eps     = gd_params['eps']      # Termination Check Tolerance

    # Line 3.- Variables
    v = 0.0
    T = 0
    k = 0
    rho_h = 0
    s = y = v = 0
    S = Y = []
    Theta = []
    # Line 4
    for t in range(N):
        # Line 5
        # p = np.zeros(theta.shape)
        s, y, v, rho, R = getStep(
            # Parámetros del algoritmo
            s, v, delta, eta, mu, alpha_s,
            # Parámetros de la función objetivo y gradiente
            fun, grad, theta, f_params
        )
        # Line 6
        thetaNew = theta + s

        # Line 7
        if (exitCheck(grad, theta, f_params, eps) == 1): break
        # Line 8
        delta, rho_k, T = getRadius(delta, s, rho, zeta, rho_k, T)
        # Line 9
        S, Y = updateSY(S, Y, s, y, rho, tau, mmr_h)
        # Line 10
        n, fK, zeta = getBatch(n, N, fK, k, K, R, gamma_1, gamma_2, zeta, fun, grad, theta, f_params)
        # Update variables
        theta = thetaNew
        Theta.append(theta)
        k = k + 1
    return np.array(Theta)

# Algorithm 4 Search direction computation algorithm
# Griffin, et. al., p. 6
def getStep(
        # Parámetros del algoritmo
        s, v, delta, eta, mu, alpha_s,
        # Parámetros adicionales del algoritmo
        Eps, S, Y,
        # Parámetros de la función objetivo y gradiente
        fun, grad, theta, f_params):
    '''
        Parameters:
            s: vector
            v: vector
            delta: radius
            eta: threshold
            mu: momentum
            alpha_s: learning rate
            eps: epsilon
            grad_k: batch gradient
            S,Y: curvature pairs
    '''
    # Line 2.- Compute batch gradient gk = \nabla f(wk)
    gk = grad(theta, f_params)
    # Line 3.- Compute p* by applying the algorithm proposed in [8] using latest S, Y pairs
    # TODO Implement calculation of p*
    ps = calculate_pstar(S,Y)
    # Line 4
    v = mu * v - eta * alpha_s * gk + (1 - eta)*s
    # Line 5
    v = min(1, delta/np.linalg.norm(v))*v
    # Line 6
    p = (1 - eta)*ps + mu * v
    # Line 7
    p = min(1, delta/np.linalg.norm(p))*p
    # Line 8
    if p.T.dot(gk) > 0:
        # Line 9
        p = -p
    gkTp = gk.T.dot(p)
    # Line 11
    if min(np.abs(gkTp),np.abs(gkTp)/(np.linalg.norm(p)**2)) < Eps*(np.linalg.norm(gk)**2):
        # Line 12.- restart LSR1 with initial matrix being I
        # Line 13.- return and restart getStep
        return
    # Line 15.- Get step-size alpha from line-search on current batch
    alpha = 1
    c1 = 1e-4
    rho_ls = 0.5
    # Line search on Armijo condition f(wk + alpha*p) <= f(wk) + c1*alpha*( grad(wk).T @ p)
    while fun(theta + alpha * p, f_params) > fun(theta, f_params) + c1 * alpha * (gk.T.dot(p)):
        alpha = rho_ls * alpha
    # Line 16
    s = alpha * p
    # Line 17
    # Compute actual reduction
    R = fun(theta + ps, f_params) - fun(theta, f_params)
    # Compute predicted reduction
    Lp = np.zeros((Y.shape[1], Y.shape[1]))
    for ii in range(Y.shape[1]):
        for jj in range(0, ii):
            Lp[ii, jj] = S[:, ii].dot(Y[:, jj])
    tmpp = np.sum((S * Y), axis=0)
    Dp = np.diag(tmpp)
    Mp = (Dp + Lp + Lp.T)
    Minvp = np.linalg.inv(Mp)
    tmpp1 = np.matmul(Y.T, ps)
    tmpp2 = np.matmul(Minvp, tmpp1)
    Bk_ps = np.matmul(Y, tmpp2)
    Q = -(gk.T.dot(ps) + 0.5 * ps.T.dot(Bk_ps))  # Compute predicted reduction
    # Line 18
    rho = R / Q
    # Line 19
    y = grad(theta + s, f_params) - gk
    return s, y, v, rho, R

# Algorithm 2 Termination Check
# Griffin, et. al., p. 5
def exitCheck(grad, theta, f_params, eps):
    # Line 3
    # TODO Implement batch gradient
    if np.linalg.norm(grad(theta, f_params)) <= eps:
        # Line 4, 5.- Evaluate new gk over whole data set
        if np.linalg.norm(grad(theta, f_params)) <= eps:
            # Line 6: Stop
            return 1
    # Line 9: Not stop
    return 0

# Algorithm 5 Example update radius function
# Griffin, et. al. p. 7
def getRadius(delta, s, rho, zeta, rho_h, T):
    # Line 2.- update non-monotone ratio threshold rho_h
    # Line 3
    rho_h = zeta * T * rho_h + rho
    # Line 4
    T = zeta * T + 1
    # Line 5
    rho_h = rho_h / T
    ns = np.linalg.norm(s)
    # Line 6
    if rho_h < 0.1:
        # Line 7
        delta = min(delta, ns)
    else:
        if rho_h >= 0.5 and ns >= delta:
            delta = 2 * delta
    return delta, rho_h, T

# Algorithm 3 Pair selection algorithm
# Griffin, et. al. p. 6
def updateSY(S, Y, s, y, rho, tau, k, mmr, mmr_h):
    # Line 2
    if rho < tau and mmr_h >= 0:
        if mmr_h == mmr:
            # Line 4.- Warm start according to [1]
            # Line 5: Generate m pairs and store in S and Y
            # TODO Implement generation of m pairs
            pass
        # Line 6
        else:
            # Line 6
            if mmr_h == 0:
                # Line 7
                S = []
                Y = []
                # Restart
    else:
        # Classic S, Y update
        # Line 11
        S.append(s)
        Y.append(y)
        # Line 12
        if k > mmr:
            S.pop(0)
            Y.pop(0)
    # Line 14
    return S,Y

# Algorithm 6 Example batch-size correction function
# Griffin, et. al. p. 7
def getBatch(nb, N, fK, k, K, R, gamma_1, gamma_2, zeta, fun, grad, theta, f_params):
    # Line 2
    if K % k == 0:
        # Line 3.- Randomly draw new progress check sample of size nh = min(1.1*nb, 5000)
        # Line 4.- Define f using the new sample
        # Line 5
        sumRj = 0
        for j in range(K-k,k):
            sumRj = sumRj + R[j]
        if grad(theta, f_params) - fK >= -gamma_1 + gamma_2 * sumRj:
            # Line 6.- Progress worse than simple line-search
            # Line 7
            nb = min(2*nb, N)
            # Line 8
            if nb == N:
                # Line 9
                zeta = 0
        # Line 12.- Update new target bound fK
        # Line 13
        fK = fun(theta, f_params)
    # Line 15.- Randomly draw batch Ik+1 with size nb defining f^
    return nb, fK, zeta



# Algorithm 2: Orthonormal Basis SR1 Method
def Orthonormal_Basis(S,Y, gamma):
    Psi = Yk - B0 @ Sk
    # 1.- Compute the Cholesky factor R of Psi' * Psi
    R = np.linalg.cholesky(Psi.T @ Psi)
    Rinv = np.linalg.inv(R)
    # Requerimos construir la matriz M para el paso siguiente
    tmpp = np.sum((S * Y), axis=0)
    Dp = np.diag(tmpp)
    Mp = (Dp + R + R.T)
    Minvp = np.linalg.inv(Mp)
    # 2.- Compute the spectral decomposition R*M*R' = U*Â*U' (solo usamos los eingenvalores)
    A, U = np.linalg.eig(R @ (Minvp @ R.T))
    A1 = A + gamma * np.eye(len(A))
    lambda_1 = np.min(A1)
    lambda_min = min(lambda_1, gamma)
    PsiRinvU = Psi @ (Rinv @ U)
    gb = PsiRinvU.T @ gk
    if lambda_min > 0 and psi(0) >= 0:
        sigma_s = 0
        Bkinv = np.linalg.inv(Bk)
        ps = - Bkinv @ gk
    else:
