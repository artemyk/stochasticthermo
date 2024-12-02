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
		
		v1=st.get_epr_ex_ig(R,p)
		v2=st.get_epr_ex_ig2(R,p)
		assert(np.isclose(v1,v2))

		v3=st.get_epr_ex_ig(R,p,cache=True)
		assert(np.isclose(v1,v3))
		v4=st.get_epr_ex_ig(R,p,cache=True)
		assert(np.isclose(v1,v4))



	def test_wasserstein(self):
		R = np.array([[-.8, 0.1, 0.2],
		              [0.3, -0.7, 0.4],
		              [0.5, 0.6, -0.6]])
		p = np.array([0.2, 0.5, 0.3])

		opt_j, opt_val = st.get_wasserstein1_speed_primal(R,R@p)
		assert(np.isclose(opt_val, 0.22))
		opt_pot, opt_val = st.get_wasserstein1_speed(R,R@p)
		assert(np.isclose(opt_val, 0.22))


	def test_cycle_decomposition(self):
		N=10
		R=st.get_random_unicyclic_ratematrix(N)
		a,b,c= st.uniform_cycle_decomposition(R)

		assert(np.all(np.array(a)>0))
		assert(np.all(np.array(b)>0))
		assert(len(c)==1)
		assert(c[0]==[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0] or c[0]==[0, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0])


	def test_darzin_inverse(self):
		X = st.get_drazin_inverse(np.array([[1,2,0],[5,1,3],[0,0,0]]))
		assert(np.allclose(X, np.array([[-1/9,2/9,-4/27],[5/9,-1/9,11/27],[0,0,0]])))


if __name__ == '__main__': 
    unittest.main() 