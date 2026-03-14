import unittest

import numpy as np

import stochasticthermo as st


class TestRB(unittest.TestCase):

    def test_get_stationary(self):
        W = np.array([[-1, 2], [1, -2]])

        p_st = st.get_stationary(W)

        self.assertTrue(np.allclose(p_st, np.array([2.0 / 3.0, 1.0 / 3.0])))

    def test_get_EPR_ratematrix(self):
        W = st.get_random_ratematrix(N=10)
        p_st = st.get_stationary(W)
        st.get_epr(W, p_st)

    def test_get_EPR_transitionmatrix(self):
        W = st.get_random_transitionmatrix(N=10)
        p_st = st.get_stationary(W, is_transition=True)
        st.get_epr(W, p_st, is_transition=True)

    def test_get_stationary_transition(self):
        N = 10
        W = st.get_random_ratematrix(N)
        T = np.eye(10) + W / np.max(-np.diag(W))

        self.assertTrue(st.is_valid_transitionmatrix(T))
        p_st = st.get_stationary(T, is_transition=True)
        self.assertTrue(np.allclose(p_st, T @ p_st))
        st.is_valid_probability(p_st)

    def test_get_random_ratematrices(self):
        st.get_random_ratematrix(N=10)
        st.get_random_ratematrix(N=10, exp=2)
        st.get_random_ratematrix(N=10, density=0.1)

    def test_get_random_transitionmatrix_sparse(self):
        np.random.seed(1)
        for density in [0.0, 0.05, 0.1]:
            T = st.get_random_transitionmatrix(N=5, density=density)
            self.assertTrue(np.all(np.isfinite(T)))
            self.assertTrue(st.is_valid_transitionmatrix(T))

    def test_get_random_1D_ratematrix(self):
        W = st.get_random_1D_ratematrix(N=10)
        self.assertEqual(W.shape, (10, 10))
        self.assertEqual(W[-1, -1], 0.0)

    def test_get_1D_ratematrix(self):
        W = st.get_1D_ratematrix([1, 2, 3], [2.1, 0.2, 0.4])
        self.assertTrue(np.allclose(W.sum(axis=0), 0))
        self.assertEqual(W[-1, -1], 0.0)
        self.assertTrue(np.allclose(st.get_stationary(W), np.array([0.0, 0.0, 0.0, 1.0])))

    def test_get_unicyclic(self):
        W = st.get_unicyclic_ratematrix([1, 2, 3], [2, 3, 5])
        self.assertTrue(st.is_valid_ratematrix(W))

    def test_get_unicyclic_two_state(self):
        W = st.get_unicyclic_ratematrix([1, 2], [3, 4])
        expected = np.array([[-4.0, 6.0], [4.0, -6.0]])
        self.assertTrue(np.allclose(W, expected))
        self.assertTrue(st.is_valid_ratematrix(W))

    def test_invalid_transitionmatrix_rejected(self):
        T = np.array([[1.2, -0.2], [-0.2, 1.2]])
        with self.assertRaises(AssertionError):
            st.is_valid_transitionmatrix(T)

    def test_numerical_radius(self):
        N = 10
        A = np.random.random((N, N))
        A = A + A.T
        max_ev = np.abs(np.linalg.eigvals(A)).max()
        self.assertTrue(np.isclose(max_ev, st.numerical_radius(A)))

        A = A - A.T
        max_ev = np.abs(np.linalg.eigvals(A)).max()
        self.assertTrue(np.isclose(max_ev, st.numerical_radius(A)))

        A = np.random.random((N, N)) + np.random.random((N, N)) * 1j
        A = A + A.conj().T
        max_ev = np.abs(np.linalg.eigvals(A)).max()
        self.assertTrue(np.isclose(max_ev, st.numerical_radius(A)))

        A = np.array([[1 / 2, 3], [0, -2]])
        self.assertTrue(np.isclose(np.round(st.numerical_radius(A), 2), 2.70))

    def test_eigenvalues(self):
        N = 10
        A = np.random.random((N, N))

        st.get_nth_eigs(A, n=1, checks=True)
        st.get_nth_eigs(A, n=2, checks=True)
        st.get_second_eigs(A, checks=True)

    def test_null_space_qr_rectangular(self):
        A = np.array([[1.0, 2.0, 3.0], [2.0, 4.0, 6.0]])
        basis = st.null_space_qr(A)
        self.assertEqual(basis.shape, (3, 2))
        self.assertTrue(np.allclose(A @ basis, 0))
        self.assertTrue(np.allclose(basis.T @ basis, np.eye(2)))

    def test_get_epr_ig(self):
        R = st.get_unicyclic_ratematrix([1, 2, 3, 4, 5], [0.1, 0.2, 0.3, 0.4, 0.1])
        N = R.shape[0]
        p = np.random.random(N)
        p /= p.sum()

        v1 = st.get_epr_ex_ig(R, p)
        v2 = st.get_epr_ex_ig2(R, p)
        self.assertTrue(np.isclose(v1, v2))

        v3 = st.get_epr_ex_ig(R, p, cache=True)
        self.assertTrue(np.isclose(v1, v3))
        v4 = st.get_epr_ex_ig(R, p, cache=True)
        self.assertTrue(np.isclose(v1, v4))

    def test_wasserstein(self):
        R = np.array(
            [[-0.8, 0.1, 0.2],
             [0.3, -0.7, 0.4],
             [0.5, 0.6, -0.6]]
        )
        p = np.array([0.2, 0.5, 0.3])

        _, opt_val = st.get_wasserstein1_speed_primal(R, R @ p)
        self.assertTrue(np.isclose(opt_val, 0.22))
        _, opt_val = st.get_wasserstein1_speed(R, R @ p)
        self.assertTrue(np.isclose(opt_val, 0.22))

    def test_cycle_decomposition(self):
        N = 10
        R = st.get_random_unicyclic_ratematrix(N)
        a, b, c = st.uniform_cycle_decomposition(R)

        self.assertTrue(np.all(np.array(a) > 0))
        self.assertTrue(np.all(np.array(b) > 0))
        self.assertEqual(len(c), 1)
        self.assertIn(
            c[0],
            [
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0],
                [0, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
            ],
        )

    def test_darzin_inverse(self):
        X = st.get_drazin_inverse(np.array([[1, 2, 0], [5, 1, 3], [0, 0, 0]]))
        self.assertTrue(
            np.allclose(X, np.array([[-1 / 9, 2 / 9, -4 / 27], [5 / 9, -1 / 9, 11 / 27], [0, 0, 0]]))
        )


if __name__ == "__main__":
    unittest.main()
