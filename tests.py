import stochasticthermo as st
import numpy as np
import unittest

class TestRB(unittest.TestCase):

	def test_get_stationary(self):
		W = np.array([[-1, 2], [1,-2]])

		p_st = st.get_stationary(W)

		assert(np.allclose(p_st, np.array([2./3., 1./3.])))


	def test_get_random_ratematrices(self):
		W = st.get_random_ratematrix(N=10)
		W = st.get_random_ratematrix(N=10, exp=2)
		W = st.get_random_ratematrix(N=10, density=0.1)

	def test_get_random_1D_ratematrix(self):
		W = st.get_random_1D_ratematrix(N=10)

	def test_get_1D_ratematrix(self):
		W = st.get_1D_ratematrix([1,2,3],[2.1,.2,.4])
		

	def test_get_random_ratematrices_with_stationary(self):

		W2 = np.array([[-1, 2,  1], 
			           [1,-3,  1], 
			           [0, 1, -2]])
		p_st2 = st.get_stationary(W2)

		W3= st.get_random_ratematrix(N=3, p_st=p_st2)

	def test_get_unicyclic(self):
		st.get_unicyclic_ratematrix([1,2,3],[2,3,5])



	def test_numerical_radius(self):
		# Test numerical radius

		N = 10
		A = np.random.random((N,N))
		A = A+A.T
		max_ev = np.abs(np.linalg.eigvals(A)).max()
		assert(np.isclose(max_ev, st.numerical_radius(A)))


	def test_eigenvalues(self):
		# Test eigenvalue code
		N = 10
		A = np.random.random((N,N))

		st.get_nth_eigs(A, n=1, checks=True)
		st.get_nth_eigs(A, n=2, checks=True)
		st.get_second_eigs(A, checks=True)


	def test_get_epr_ig(self):
		R=st.get_unicyclic_ratematrix([1,2,3,4,5],[.1,.2,.3,.4,.1])
		N=R.shape[0]
		p=np.random.random(N) ; p/=p.sum()
		st.get_epr_ex_ig(R,p)


	def test_wasserstein(self):
		R = np.array([[-.8, 0.1, 0.2],
		              [0.3, -0.7, 0.4],
		              [0.5, 0.6, -0.6]])
		p = np.array([0.2, 0.5, 0.3])

		opt_j, opt_val = st.get_wasserstein1_speed(R,p)
		assert(np.isclose(opt_val, 0.22))



if __name__ == '__main__': 
    unittest.main() 