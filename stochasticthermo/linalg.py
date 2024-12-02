import numpy as np
import scipy.linalg


def get_nth_eigs(A, n=0, checks=False):
    """
    Returns n-th eigenvalue and eigenvectors of matrix A
    (sorted in ascending order by real part)

    Parameters
    ----------
    A : NxN np.array
       Matrix of interest
    n : int (default 1)
       Return n-th lowest eigenvalue 
    checks : bool (False)
        Run some sanity checks


    Returns
    -------
    eigenvalue : float
        Lowest eigenvalue
    u : np.array
        Left eigenvector
    v : np.array
        Right eigenvector
    """

    d, u, v = scipy.linalg.eig(A, left=True, right=True)
    ix = np.argsort(np.real(d))[n]
    ll = u[:,ix] / np.linalg.norm(u[:,ix])
    rr = v[:,ix] / np.linalg.norm(v[:,ix])
    ev = d[ix]

    if False: # do checks
        assert(np.allclose((ll@A), np.conj(ev)*ll))
        assert(np.allclose((A@rr), ev*rr))

    return ev, ll, rr

    

def get_second_eigs(A, checks=False):
    """
    Returns second-largest eigenvalue and eigenvectors of matrix A

    Parameters
    ----------
    A : NxN np.array
       Matrix of interest
    checks : bool (False)
        Run some sanity checks

    Returns
    -------
    eigenvalue : float
        Second eigenvalue
    u : np.array
        Left eigenvector
    v : np.array
        Right eigenvector
    """
    return get_nth_eigs(A, n=-2, checks=checks)



def numerical_radius(A):
    """
    Computes the numerical radius of a matrix, 
        w(A) = max { |x^* A x| : x ∈ C^n }
    Using algorithm from https://nhigham.com/2023/07/11/what-is-the-numerical-range-of-a-matrix/
    """
    import cvxpy as cp
    
    Z = cp.Variable(A.shape,symmetric=True)
    M = cp.vstack([cp.hstack([Z,A]),cp.hstack([A.T, -Z])])
    
    obj = cp.lambda_min(M)
    
    prob = cp.Problem(-cp.Maximize(obj))
    return prob.solve()




def null_space_qr(A, tol=1e-12):
    Q, R = scipy.linalg.qr(A.T, mode='economic')
    
    # Determine the rank of A using the diagonal of R
    rank = np.sum(np.abs(np.diag(R)) > tol)
    
    # Null space basis vectors are in Q[:, rank:]
    null_space = Q[:, rank:]
    
    return null_space
