# stochasticthermo
Stochastic thermodynamics of discrete systems for Python.

The package uses the convention `W[i, j]` for the transition rate from state `j` to state `i`, so valid rate matrices have non-positive diagonals, non-negative off-diagonals, and each column sums to zero. Transition matrices follow the same column-stochastic convention and each column sums to one.

You can install this by running
```
python -m pip install https://github.com/artemyk/stochasticthermo/archive/refs/heads/main.zip
```

If you would like to install an editable development version, you can run
```
python -m pip install --editable .
```
in the downloaded directory.

## Notes

- `get_random_transitionmatrix()` now guarantees a valid column-stochastic matrix even for very sparse densities by inserting a self-loop when a sampled column would otherwise be empty.
- `get_unicyclic_ratematrix()` supports the `N=2` case by combining the forward and backward contributions into the single edge between the two states.
- `get_1D_ratematrix()` and `get_random_1D_ratematrix()` construct an open chain with an absorbing right boundary. For periodic chains, use `get_unicyclic_ratematrix()` instead.
