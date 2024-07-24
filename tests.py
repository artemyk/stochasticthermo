import stochasticthermo as st
import numpy as np


W = np.array([[-1, 2], [1,-2]])

p_st = st.get_stationary(W)

assert(np.allclose(p_st, np.array([2./3., 1./3.])))



W2 = np.array([[-1, 2,  1], 
	           [1,-3,  1], 
	           [0, 1, -2]])

p_st2 = st.get_stationary(W2)


W = st.get_random_ratematrix(N=10)
W = st.get_random_ratematrix(N=10, exp=2)
W = st.get_random_ratematrix(N=10, density=0.1)

W3= st.get_random_ratematrix(N=3, p_st=p_st2)


st.get_unicyclic_ratematrix([1,2,3],[2,3,5])



# Test numerical radius

N = 10
A = np.random.random((N,N))
A = A+A.T
max_ev = np.abs(np.linalg.eigvals(A)).max()
assert(np.isclose(max_ev, st.numerical_radius(A)))
