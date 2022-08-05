import numpy as np
import scipy.linalg

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


def get_second_eigs(W):
    # return second eigenvalue and eigenvectors
    d, u, v = scipy.linalg.eig(W, left=True, right=True)
    ix = np.argsort(np.real(d))[-2]
    l2 = u[:,ix] / np.linalg.norm(u[:,ix])
    r2 = v[:,ix] / np.linalg.norm(v[:,ix])
    ev = d[ix]

    if False: # do checks
        assert(np.allclose((l2@W), np.conj(ev)*l2))
        assert(np.allclose((W@r2), ev*r2))

    return ev, l2, r2

def get_random_ratematrix(N,p=None, exp=1):
    # Generate random rate matrix. 
    # N is number of states
    # p is stationary distribution (if we want to normalize activity to 1)
    # Increase exp to make distribution fatter tailed
    W = np.random.random((N,N))**exp  # make distribution fatter tailed
    np.fill_diagonal(W,0)
    
    if p is not None:
        assert(len(p)==N)
        fluxes = W*p[None,:]    
        W     /= fluxes.sum()                    # normalize activity to 1

    W     -= np.diag(W.sum(axis=0))
    return W


def get_random_ratematrix_pareto(N):
    R = np.random.pareto(.75, (N, N))
    #R = np.random.random((N,N))
    np.fill_diagonal(R,0)
    R -= np.diag(np.sum(R,axis=0))
    return R



