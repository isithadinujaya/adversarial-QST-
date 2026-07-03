# Validation performed

The supplied repository was checked with:

```bash
pytest -q
```

All six core tests passed. The tests cover:

- Pauli-cube dimensions and per-setting normalization for one, two, and three qubits;
- Cholesky output physicality;
- the trace-distance physical attack constraint;
- the product-simplex and `l_infinity` PGD projection;
- backward propagation through the latest fidelity loss;
- fidelity of a state with itself.

A complete one-qubit smoke training run, checkpoint save/load, evaluation over all attack branches, all sweep scripts on reduced grids, and all result/method figure generators were also executed successfully.

Direct forward checks were run for the one-, two-, and three-qubit MLP configurations. Their outputs had shape `2x2`, `4x4`, and `8x8`, unit trace, and nonnegative eigenvalues up to numerical precision.
