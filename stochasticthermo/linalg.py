import numpy as np

def numerical_radius(A):
    import cvxpy as cp
    
    Z = cp.Variable(A.shape,symmetric=True)
    M = cp.vstack([cp.hstack([Z,A]),cp.hstack([A.T, -Z])])
    
    obj = cp.lambda_min(M)
    
    prob = cp.Problem(-cp.Maximize(obj))
    return prob.solve()

