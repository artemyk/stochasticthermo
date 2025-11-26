import numpy as np
from .ratematrix import get_stationary, is_valid_ratematrix, is_valid_transitionmatrix, is_valid_probability

def get_epr(W,p,checks=True, is_transition=False): 
    # get entropy production rate incurred by rate matrix W on distribution p
    if checks:
        is_valid_probability(p)
        if is_transition:
            is_valid_transitionmatrix(W)
        else:
            is_valid_ratematrix(W)

    N   = len(p)
    fluxes = W*p[None,:]
    r   = 0
    for i in range(N):
        for j in range(N):
            if i != j and fluxes[i,j] != 0:
                assert(fluxes[i,j]>0)
                r +=  fluxes[i,j]*np.log( fluxes[i,j]/fluxes[j,i] )
    return r

def get_entropy_change(W,p): 
    # get dS/dt by rate matrix W on distribution p, where S is Shannon entropy (in bits)
    N   = len(p)
    dp  = W@p

    assert(np.isclose(dp.sum(),0))

    ixs = dp != 0
    return -dp[ixs]@np.log(p[ixs])



def get_epr_ex_hs(W, p, checks=True): 
    # Calculate nonadiabatic (excess) EP rate of Hatano-Sasa
    # W is rate matrix, p is distribution over states p(x)

    dp = W.dot(p)
    st = np.ravel(get_stationary(W, checks=checks))
    return -dp.dot(np.log(p/st))


def get_epr_ex_ons(W,p, S=None):
    # Onsager projective excess EP rate
    # See Yoshimura et al., https://arxiv.org/abs/2205.15227
    # Parameters
    # W : (NxN)   rate matrix
    # p : (N,1)   probability distribution
    # S : (N,NxN) stoichimetric matrix
    N = len(p)
    
    if S is None: # stoichometric matrix not passed in
        S = np.zeros((N,N*N))
        for i in range(N):
            for j in range(N):
                if i != j:
                    S[i,i*N+j] =  1
                    S[j,i*N+j] = -1

    fluxes = W*p[None,:]   # fluxes[j,i] is flux from i to j
    np.fill_diagonal(fluxes, 0)
    fluxes_flat     = np.ravel(fluxes)    + 1e-20
    fluxes_rev_flat = np.ravel(fluxes.T)  + 1e-20
    forces          = np.log( fluxes_flat/fluxes_rev_flat)
    
    # 1/2 factor is because we count reversible reactions as two edges. 
    # See discussion around Eq. (A.2) in https://arxiv.org/pdf/2206.14599.pdf 
    L               = (1/2) * np.diag( (fluxes_flat-fluxes_rev_flat) / (1e-20+forces) )
    phi             = np.linalg.lstsq( S @ L @ S.T , S @ L @ forces, rcond=None)[0]
    proj            = S.T @ phi
    # mx            = S.T @ np.linalg.pinv(S @ L @ S.T, hermitian=True) @ S @ L
    # proj2 = mx.dot(forces)
    # assert(np.allclose(proj, proj2))
    ons_ep = proj.dot(L).dot(proj)
    return ons_ep, L


_get_epr_ex_ig_cache = {}

def get_epr_ex_ig(W, p, return_optimal_potential=False, cache=False): 
    # !TODO! Convert to torch LBGFS optimization, its probably much faster.
    # 
    # Calculate excess EP rate based on information-geometric projection
    # W is rate matrix, p is distribution over states p(x)
    # Here we solve the dual problem (Legendre transform)
    # See Kolchinsky et al., https://arxiv.org/abs/2206.14599
    import cvxpy as cp


    is_valid_ratematrix(W)
    N  = W.shape[0]

    prob       = None
    mask_pos   = np.ones((N,N)) - np.eye(N)
    cur_fluxes = W*p[None,:]

    if cache:
        if N in _get_epr_ex_ig_cache:
            obj, x, fluxes, prob = _get_epr_ex_ig_cache[N]
        else:
            fluxes = cp.Parameter((N,N), nonneg=True)

        fluxes.value = cur_fluxes*mask_pos
    else:
        fluxes       = cur_fluxes*mask_pos
        
    if prob is None:
        x  = cp.Variable(shape=N)
        dp = (fluxes-fluxes.T).sum(axis=1)

        obj = -dp@x
    
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                if not cache and np.isclose(fluxes[j,i],0):
                    continue
                obj -= fluxes[j,i]*(cp.exp(x[j]-x[i])-1)
                
        prob = cp.Problem(cp.Maximize(obj))

    if cache and N not in _get_epr_ex_ig_cache:
        _get_epr_ex_ig_cache[N] = (obj, x, fluxes, prob)

    prob.solve(solver=cp.CLARABEL ) # , max_iter=1500) 

    if return_optimal_potential:
        return obj.value, x.value
    else:
        return obj.value

def get_TSL_flow(W,p):
    import cvxpy as cp

    # Stoichiometric matrix
    S = []
    fluxes = []
    n = W.shape[0]
    
    idxs = {}
    for i in range(n):
        for j in range(n):
            if i != j and not np.isclose(W[i, j], 0):
                r = np.zeros(n)
                r[i]=1
                r[j]=-1
                S.append(r)
                idxs[(i,j)] = len(fluxes)
                fluxes.append(p[j]*W[i,j])
                
    S = np.array(S)
    fluxes = np.array(fluxes)
    # Build undirected edge list
    pairs = []
    edge_idx = {}
    for (i,j) in idxs:
        if (j,i) in idxs and (j,i) not in edge_idx:
            edge_idx[(i,j)] = len(pairs)
            pairs.append((i,j))

    E = len(pairs)
    # Incidence B (n x E)
    B = np.zeros((n, E))
    for k, (i,j) in enumerate(pairs):
        B[i, k] = 1.0
        B[j, k] = -1.0

    d = S.T @ fluxes                     # node imbalances of original fluxes
    Btot = fluxes.sum()

    # LP: min ||z||_1 s.t. B z = d
    z = cp.Variable(E)
    t = cp.Variable(E, nonneg=True)
    constraints = [B @ z == d, -t <= z, z <= t]
    lp = cp.Problem(cp.Minimize(cp.sum(t)), constraints)
    lp.solve(solver=cp.CLARABEL)

    Lstar = t.value.sum()
    rho = Lstar / Btot
    rstar = (1 + rho) / (1 - rho) if rho < 1 else np.inf    # rho<1 in feasible cases
    # reconstruct b*
    b = np.zeros(len(fluxes))
    for (i,j), k in edge_idx.items():
        zij = z.value[k]
        if zij >= 0:
            b[idxs[(i,j)]] = rstar/(rstar-1) * zij
            b[idxs[(j,i)]] = 1/(rstar-1) * zij
        else:
            zij = -zij
            b[idxs[(i,j)]] = 1/(rstar-1) * zij
            b[idxs[(j,i)]] = rstar/(rstar-1) * zij
    print(lp.status, np.log(rstar)*lp.value)
    return Btot
    
def get_info_TSL(W, p, version=1):
    import cvxpy as cp

    # Stoichiometric matrix
    S = []
    fluxes = []
    n = W.shape[0]
    
    idxs = {}
    for i in range(n):
        for j in range(n):
            if i != j and not np.isclose(W[i, j], 0):
                r = np.zeros(n)
                r[i]=1
                r[j]=-1
                S.append(r)
                idxs[(i,j)] = len(fluxes)
                fluxes.append(p[j]*W[i,j])
                
    S = np.array(S)
    fluxes = np.array(fluxes)

    m = len(fluxes)
    b = cp.Variable(m, nonneg=True)
    constraints = [  cp.sum(b) <= fluxes.sum(), S.T @ b == S.T @ fluxes, ]
    eps = 1e-30
    if version == 1:
        a = cp.Variable(m, nonneg=True)
        constraints.append( S.T @ a == -S.T @ b )
        objective = cp.sum(cp.kl_div(a + eps, b + eps))
    else:
        objective = 0
        for i in range(n):
            for j in range(n):
                if (j,i) in idxs:
                    objective += cp.kl_div(b[idxs[(j,i)]] + eps, b[idxs[(i,j)]] + eps)

    prob = cp.Problem(cp.Minimize(objective), constraints)
    prob.solve(verbose=False, solver=cp.CLARABEL)


    #print("status:", prob.status)
    #print("optimal D:", prob.value)
    #print("b*:", b.value)
    print(cp.sum(b).value, fluxes.sum())

    W_opt = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            if (i,j) in idxs:
                W_opt[i,j] = b.value[idxs[(i,j)]] / p[j]
                W_opt[j,j] -= W_opt[i,j]
    return prob.value, W_opt


    
def get_epr_ex_ig2(W,p, return_optimal_potential=False): 
    # !TODO! Convert to torch LBGFS optimization, its probably much faster.
    # 
    # Calculate excess EP rate based on information-geometric projection
    # W is rate matrix, p is distribution over states p(x)
    # Here we solve the primal problem
    import cvxpy as cp
    
    fluxes = W*p[None,:]   # fluxes[j,i] is flux from i to j
    np.fill_diagonal(fluxes, 0)
    N = len(p)
    x = cp.Variable(shape=fluxes.shape, nonneg=True)
    c = [0,]* N
    f = []

    dp  = W@p
    assert(np.isclose(dp.sum(),0))

    obj = 0
    for i in range(N):
        for j in range(N):
            if i == j: 
                continue
            c[j] += x[j,i] - x[i,j]
            if fluxes[j,i] == 0:
                f.append( x[i,j] == 0 )
            else:
                obj += cp.kl_div(x[i,j], fluxes[j,i])
            
    prob = cp.Problem(cp.Minimize(obj), [c[i] == dp[i] for i in range(N)] + f)
    prob.solve(solver=cp.CLARABEL) # , max_iter=500)
    
    if return_optimal_potential:
        return obj.value, x.value
    else:
        return obj.value    
    
