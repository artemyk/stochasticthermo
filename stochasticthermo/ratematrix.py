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

    N = scipy.linalg.null_space(W)
    if checks and N.shape[1] != 1:
        rnk = N.shape[1]
        raise Exception(f'rank of null space not 1. {rnk}')
        
    st          = N.flatten()

    if checks:
        assert(near_zero(np.imag(st)))

    st          = np.real(st)
    st         /= st.sum()
    
    if st.min() < -1e-10:
        raise Exception(f'Some stationary probabilities are negative {st}')
    st[st<0] = 0
        
    if checks:
        assert(near_zero(W @ st))

    return st


def get_second_eigs(W):
    """
    Returns second eigenvalue and eigenvectors of matrix W

    Parameters
    ----------
    W : NxN np.array

    Returns
    -------
    eigenvalue : float
        Second eigenvalue
    u : np.array
        Left eigenvector
    v : np.array
        Right eigenvector
    """

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
    exp  : float (default 1)
        parameter to make distribution fatter tailed
    """

    W = np.random.random((N,N))**exp  # make distribution fatter tailed
    if density != 1:
        mask = (np.random.random((N,N))<density).astype('float')
        W = W*mask

    np.fill_diagonal(W,0)
    W  -= np.diag(W.sum(axis=0))
    return W


def get_fluxes(W, p=None, checks=True):
    """
    Get matrix of 1-way fluxes given rate matrix W and distribution p
    If p is not specified, use the steady state distribution

    """
    if checks:
        assert(near_zero(W.sum(axis=0)))

    if p is None:
        p = get_stationary(W, checks=checks)
    fluxes = W*p[None,:]
    return fluxes


def get_dynamical_activities(W, p=None, checks=True):
    """
    Get matrix of dynamical activities given rate matrix W and 
    distribution p
    If p is not specified, use the steady state distribution
    """
    
    fluxes = get_fluxes(W, p, checks=checks)
    np.fill_diagonal(fluxes, 0)
    return fluxes + fluxes.T


def get_random_ratematrix_pareto(N):
    R = np.random.pareto(.75, (N, N))
    #R = np.random.random((N,N))
    np.fill_diagonal(R,0)
    R -= np.diag(np.sum(R,axis=0))
    return R


def get_adjoint_ratematrix(W, checks=True):
    # Return adjoint W_ji^* = W_ij pi_j / pi_i
    st = get_stationary(W, checks=checks)
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
    assert(np.min(forward_rates) >= 0 and np.min(backward_rates)>=0)
    W = np.zeros((N,N))
    for i in range(N):
        W[(i+1)%N,i] = forward_rates[i]
        W[(i-1)%N,i] = backward_rates[i]
        W[i,i]      -= forward_rates[i]+backward_rates[i]
    return W


def get_random_unicyclic_ratematrix(N, p=1.0, g=1.0, **kwargs):
    """
    Generate random unicyclic rate matrix

    Parameters
    ----------
    N : int
        number of states
    p : float (default 1)
        control homogeneity (p=0 all uniform)
    g : float (default 1)
        degree of disequilibrium (forward rates bigger than reverse)
    kwargs : dict
        additional keyword arguments to pass to get_unicyclic_ratematrix
    """

    k  = np.random.random(N)**p
    kk = g * np.random.random(N)**p
    return get_unicyclic_ratematrix(k, kk, **kwargs)


