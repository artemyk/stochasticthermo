import numpy as np
import scipy.linalg


def near_zero(x):
    return np.abs(x).max() < 1e-8


def is_valid_ratematrix(W):
    diags = np.diag(W)
    assert(np.all(diags<=0))          # diagonals are negative
    assert(np.all(W>=np.diag(diags))) # off diagonals are positive
    assert(near_zero(W.sum(axis=0)))

def is_valid_probability(p):
    assert(np.all(p>=0))
    assert(np.isclose(np.sum(p),1))

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
        is_valid_ratematrix(W)

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

    Note that the diagonals are set to 0
    """
    if checks:
        is_valid_ratematrix(W)

    if p is None:
        p = get_stationary(W, checks=checks)
    fluxes = W*p[None,:]
    np.fill_diagonal(fluxes, 0)
    
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

def get_random_birthdeath_ratematrix(N, p=1.0, g=1.0, **kwargs):
    """
    Generate random birth-death rate matrix

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
    k[-1] = kk[0] = 0
    return get_unicyclic_ratematrix(k, kk, **kwargs)





def get_1D_ratematrix(forward_rates, backward_rates):
    """
    Generate rate matrix representing 1D random walk

    Parameters
    ----------
    forward_rates : np.array of floats
        forward rates i->i+1  for i=0 to N
    backward_rates : np.array of floats
        backward rates i->i-1 for i=1 to N+1

    Returns
    -------
    (N+1)x(N+1) rate matrix, where N = len(forward_rates)
    """

    N = len(forward_rates)
    assert(N == len(backward_rates))
    assert(np.min(forward_rates) >= 0 and np.min(backward_rates)>=0)
    W = np.zeros((N+1,N+1))
    for i in range(N):
        if i <= N:
            W[i+1,i] = forward_rates[i]
        if i >  0:
            W[i-1,i] = backward_rates[i-1]
        W[i,i]      -= W[:,i].sum()
    return W

def get_random_1D_ratematrix(N, p=1.0, g=1.0):
    """
    Generate random 1D random walk rate matrix

    Parameters
    ----------
    N : int
        number of states
    p : float (default 1)
        control homogeneity (p=0 all uniform)
    g : float (default 1)
        degree of disequilibrium (forward rates bigger than reverse)
    """
    k  = np.random.random(N-1)**p
    kk = g * np.random.random(N-1)**p
    return get_1D_ratematrix(k, kk)



from scipy.optimize import linprog

def get_wasserstein1_speed_primal(R, dp):
    # Ensure the rate matrix R is valid and dp sums close to zero
    is_valid_ratematrix(R)
    assert(np.isclose(dp.sum(), 0))

    n = R.shape[0]
    
    # Define the number of variables: one for each (i, j) pair in J
    num_vars = n * n
    
    # Set up the objective to minimize the sum of entries in J
    c = np.ones(num_vars)  # We want to minimize the sum of all entries in J

    # Constraints to enforce J[i, j] == 0 where R[i, j] == 0
    A_eq = []
    b_eq = []
    
    # Populate constraints
    for i in range(n):
        # Constraint: Row sum minus column sum equals dp[i]
        row_constraint = np.zeros(num_vars)
        col_constraint = np.zeros(num_vars)
        
        for j in range(n):
            if np.isclose(R[i, j], 0):
                # If R[i, j] == 0, we set J[i, j] = 0
                zero_constraint = np.zeros(num_vars)
                zero_constraint[i * n + j] = 1
                A_eq.append(zero_constraint)
                b_eq.append(0)
                
            row_constraint[i * n + j] = 1  # J[i, j] term in row sum
            col_constraint[j * n + i] = -1 # J[j, i] term in column sum

        A_eq.append(row_constraint + col_constraint)
        b_eq.append(dp[i])

    # Convert to arrays for linprog
    A_eq = np.array(A_eq)
    b_eq = np.array(b_eq)

    # Use SciPy's linprog to solve the optimization problem
    result = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=(0, None), method='highs')
    
    if result.success:
        # Reshape the solution back into an (n x n) matrix
        J = result.x.reshape((n, n))
        return J, result.fun
    else:
        raise ValueError("Optimization did not converge")



def get_wasserstein1_speed(R, dp):
    # Ensure the rate matrix R is valid and dp sums close to zero
    is_valid_ratematrix(R)
    assert(np.isclose(dp.sum(), 0))

    n = R.shape[0]
    
    # We split each potential difference constraint |pot[i] - pot[j]| <= 1 
    # into two linear constraints: pot[i] - pot[j] <= 1 and pot[j] - pot[i] <= 1
    num_constraints = sum(1 for i in range(n) for j in range(n) if not np.isclose(R[i, j], 0))
    c = -dp  # Maximize dp @ pot can be represented as minimizing -dp @ pot in linprog

    # Define constraints
    A = np.zeros((2 * num_constraints, n))
    b = np.ones(2 * num_constraints)

    constraint_index = 0
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if not np.isclose(R[i, j], 0):
                # Constraint pot[i] - pot[j] <= 1
                A[constraint_index, i] = 1
                A[constraint_index, j] = -1
                # Constraint pot[j] - pot[i] <= 1
                A[constraint_index + 1, i] = -1
                A[constraint_index + 1, j] = 1
                constraint_index += 2

    # Use SciPy's linprog to solve
    result = linprog(c, A_ub=A, b_ub=b, method='highs')
    
    if result.success:
        return result.x, -result.fun
    else:
        raise ValueError("Optimization did not converge")

