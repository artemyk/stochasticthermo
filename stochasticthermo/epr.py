import numpy as np
from .ratematrix import get_stationary

def get_epr(W,p): 
    # get entropy production rate incurred by rate matrix W on distribution
    N   = len(p)
    fluxes = W*p[None,:]
    r   = 0
    for i in range(N):
        for j in range(N):
            if i != j:
                r +=  fluxes[i,j]*np.log( fluxes[i,j]/fluxes[j,i] )
    return r


def get_epr_ex_hs(W,p): 
    # Calculate nonadiabatic (excess) EP rate of Hatano-Sasa
    # W is rate matrix, p is distribution over states p(x)

    dp = W.dot(p)
    st = np.ravel(get_stationary(W))
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


def get_epr_ex_ig(W, p, return_optimal_potential=False): 
    # Calculate excess EP rate based on information-geometric projection
    # W is rate matrix, p is distribution over states p(x)
    # Here we solve the dual problem (Legendre transform)
    # See Kolchinsky et al., https://arxiv.org/abs/2206.14599
    import cvxpy as cp

    assert(np.allclose(W.sum(axis=0),0))
    dp = W.dot(p)
    
    fluxes = W*p[None,:]   # fluxes[j,i] is flux from i to j

    N   = len(p)
    x   = cp.Variable(shape=N)
    obj = -dp @ x 
    
    for i in range(N):
        for j in range(N):
            if i != j and fluxes[j,i] != 0:
                obj -= fluxes[j,i]*(cp.exp(x[j]-x[i])-1)
                
    prob = cp.Problem(cp.Maximize(obj))
    prob.solve(solver=cp.ECOS, max_iters=10000)
    
    if return_optimal_potential:
        return obj.value, x.value
    else:
        return obj.value
    
    
def get_epr_ex_ig2(W,p, return_optimal_potential=False): 
    # Calculate excess EP rate based on information-geometric projection
    # W is rate matrix, p is distribution over states p(x)
    # Here we solve the primal problem
    import cvxpy as cp
    
    fluxes = W*p[None,:]   # fluxes[j,i] is flux from i to j
    np.fill_diagonal(fluxes, 0)
    N = len(p)
    x = cp.Variable(shape=fluxes.shape)
    c = [0,]* N
    f = []

    dp  = W.T.dot(p)
    obj = 0
    for i in range(N):
        for j in range(N):
            c[j] += x[j,i] - x[i,j]
            if fluxes[j,i] == 0:
                f.append( x[i,j] == 0 )
            else:
                obj  += cp.kl_div(x[i,j], fluxes[j,i])
            
    prob = cp.Problem(cp.Minimize(obj), [c[i] == dp[i] for i in range(N)] + f)
    prob.solve(solver=cp.ECOS, max_iters=10000)
    
    if return_optimal_potential:
        return obj.value, x.value
    else:
        return obj.value    
    

