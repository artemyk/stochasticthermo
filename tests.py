import stochasticthermo
import numpy as np


W = np.array([[-1, 2], [1,-2]])

st = stochasticthermo.get_stationary(W)

assert(np.allclose(st, np.array([2./3., 1./3.])))



W2 = np.array([[-1, 2,  1], 
	           [1,-3,  1], 
	           [0, 1, -2]])

st = stochasticthermo.get_stationary(W2)
