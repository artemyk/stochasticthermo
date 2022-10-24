import numpy as np


def get_stationary(W): 
    assert(np.allclose(W.sum(axis=0),0))

    # get stationary distribution of rate matrix W. W[i,j] is transition rate from j->i
    evec, evals = np.linalg.eig(W)
    evec1ix     = np.isclose(evec,0)
    if not (sum(evec1ix)==1):
        raise Exception('# of eigenvalue=0 eigenvector is not 1. %s' % str(evec))
        
    st          = np.ravel(evals[:,evec1ix])
    assert(np.allclose(np.imag(st), 0))
    st          = np.real(st)
    st         /= st.sum()
    
    if not np.all(st>=0):
        raise Exception('Some stationary probabilities are negative %s' % str(st))
        
    assert(np.allclose(W @ st,0))
    return st


# def get_st(R):
#     evals, evecs = scipy.linalg.eig(R)
#     ixs          = np.flatnonzero(np.isclose(evals,0,atol=1e-6))
#     if len(ixs) != 1: raise Exception()
#     p  = evecs[:,ixs[0]]
#     if not np.allclose(R.dot(p),0)  : raise Exception()
#     if not np.allclose(np.imag(p),0): raise Exception()
#     p  = np.real(p)
#     p /= p.sum()
#     return p


def get_random_ratematrix(N,p, exp=1):
    # Generate random rate matrix. Increase exp to make distribution fatter tailed
    assert(len(p)==N)
    W = np.random.random((N,N))**exp  # make distribution fatter tailed
    np.fill_diagonal(W,0)
    
    fluxes = W*p[None,:]    
    W     /= fluxes.sum()                    # normalize activity to 1
    W     -= np.diag(W.sum(axis=0))
    return W


def get_adjoint_ratematrix(W):
    # Return adjoint W_ji = W_ij pi_j / pi_i
    st = get_stationary(W)
    n  = len(st)
    Wadj = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            Wadj[i,j] = W[j,i]*st[i]/st[j]
    return Wadj
