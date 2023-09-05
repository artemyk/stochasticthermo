import numpy as np
import scipy.linalg

eps = 1e-5

def near_zero(x):
    return np.abs(x).max() < 1e-8

def get_stationary(W, checks=True):
    """
    Get stationary distribution of rate matrix W. 

    Parameters
    ----------
    W : 2D np.array
        rate matrix, W[i,j] is transition rate from j->i
    checks : bool
        perform sanity checks on W and stationary state
    """ 
    if checks:
        assert(near_zero(W.sum(axis=0)))

    evec, evals = np.linalg.eig(W)
    evec1ix     = np.abs(evec)<=eps
    if not (sum(evec1ix)==1):
        raise Exception('# of eigenvalue=0 eigenvector is not 1. %s' % str(evec))
        
    st          = evals[:,evec1ix].T[0]

    if checks:
        assert(near_zero(np.imag(st)))

    st          = np.real(st)
    st         /= st.sum()
    
    if st.min() < -1e-10:
        raise Exception('Some stationary probabilities are negative %s' % str(st))
    st[st<0] = 0
        
    if checks:
        assert(near_zero(W @ st))

    return st


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

def get_random_ratematrix(N, density=1, p_st=None, exp=1):
    """
    Generate random rate matrix. 
    
    Parameters
    ----------
    N    : int
        number of states
    density : float (default 1)
        percentage of edges to fill in
    p_st : array of float (default None)
        desired stationary distribution (if we want to normalize activity to 1)
    exp  : float (default 1)
        parameter to make distribution fatter tailed
    """

    W = np.random.random((N,N))**exp  # make distribution fatter tailed
    if density != 1:
        mask = (np.random.random((N,N))<density).astype('float')
        W = w*mask

    np.fill_diagonal(W,0)
    
    if p is not None:
        assert(len(p)==N)
        fluxes = W*p[None,:]    
        W     /= fluxes.sum()                    # normalize activity to 1

    W     -= np.diag(W.sum(axis=0))
    return W


def get_fluxes(W, p=None):
    # Get matrix of 1-way fluxes given rate matrix W and distribution p
    # If p is not specified, use the steady state distribution
    assert(near_zero(W.sum(axis=0)))
    if p is None:
        p = get_stationary(W)
    fluxes = W*p[None,:]
    return fluxes

def get_dynamical_activities(W, p=None):
    # Get matrix of dynamical activities given rate matrix W and 
    # distribution p
    # If p is not specified, use the steady state distribution
    fluxes = get_fluxes(W, p)
    np.fill_diagonal(fluxes, 0)
    return fluxes + fluxes.T


def get_random_ratematrix_pareto(N):
    R = np.random.pareto(.75, (N, N))
    #R = np.random.random((N,N))
    np.fill_diagonal(R,0)
    R -= np.diag(np.sum(R,axis=0))
    return R


def get_adjoint_ratematrix(W):
    # Return adjoint W_ji^* = W_ij pi_j / pi_i
    st = get_stationary(W)
    n  = len(st)
    Wadj = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            Wadj[i,j] = W[j,i]*st[i]/st[j]
    return Wadj


def get_unicyclic_ratematrix(forward_rates, backward_rates):
    """
    Generate unicyclic rate matrix

    Parameters
    ----------
    forward_rates : np.array of floats
        forward rates
    backward_rates : np.array of floats
        backward rates
    """

    N = len(forward_rates)
    assert(N == len(backward_rates))
    assert(forward_rates.min() > 0 and backward_rates.min()>0)
    W = np.zeros((N,N))
    for i in range(N):
        W[(i+1)%N,i] = forward_rates[i]
        W[(i-1)%N,i] = backward_rates[i]
        W[i,i]      -= forward_rates[i]+backward_rates[i]
    return W

def get_random_unicyclic_ratematrix(N):
    k  = np.random.random(N)
    kk = np.random.random(N)
    return get_unicyclic_ratematrix(k, kk)


