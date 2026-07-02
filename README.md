# Modular adversarial quantum state tomography

This repository implements the latest agreed reconstruction-only model for one-, two-, and three-qubit quantum state tomography.

## Mathematical model implemented

For `n` qubits, `d = 2^n`. Measurements use every local Pauli setting in `{X,Y,Z}^n`. Each setting has `d` outcomes, so the network input dimension is

- one qubit: `3 × 2 = 6`,
- two qubits: `9 × 4 = 36`,
- three qubits: `27 × 8 = 216`.

For each setting, counts are sampled exactly from a multinomial distribution with the configured number of shots and normalized into a probability block.

The reconstructor predicts `d^2` real parameters. They are converted into a lower-triangular complex matrix `T`, then into a physical density matrix

`rho_hat = T T† / Tr(T T†)`.

The training loss is

`L = L_clean + L_adv + 0.1 L_cons`,

where

- `L_clean = 1 - F(rho, rho_hat_clean)`,
- `L_adv = 1 - F(rho, rho_hat_adv)`,
- `L_cons = 1 - F(stop_gradient(rho_hat_clean), rho_hat_adv)`.

Physical copy-replacement attacks use

`rho_eff = (1-alpha) rho + alpha sigma`

and enforce

`D_tr(rho, rho_eff) <= epsilon_physical`.

The code therefore uses

`alpha_effective = min(alpha_requested, epsilon_physical / D_tr(rho, sigma))`.

Frequency PGD acts directly on normalized frequency blocks and enforces both

- `||f_adv - f_clean||_infinity <= epsilon_frequency`, and
- nonnegative, separately normalized outcome probabilities for every measurement setting.

There is no attack-detection head and no BCE loss.

## Architecture modularity

Only `qst/models/mlp.py` is architecture-specific. The measurement model, state generation, attacks, Cholesky output head, losses, trainer, evaluator, and plotting code depend only on the model interface `model(frequencies) -> density_matrix`. A later CNN, residual MLP, or Transformer can be registered without changing the rest of the pipeline.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
```

## Train

```bash
python -m scripts.train --config configs/one_qubit.yaml
python -m scripts.train --config configs/two_qubit.yaml
python -m scripts.train --config configs/three_qubit.yaml
```

The best checkpoints are written to:

```text
outputs/one_qubit/best.pt
outputs/two_qubit/best.pt
outputs/three_qubit/best.pt
```

## Evaluate and run sweeps

```bash
python -m scripts.evaluate --config configs/one_qubit.yaml --checkpoint outputs/one_qubit/best.pt
python -m scripts.sweep_alpha --config configs/one_qubit.yaml --checkpoint outputs/one_qubit/best.pt
python -m scripts.sweep_epsilon --config configs/one_qubit.yaml --checkpoint outputs/one_qubit/best.pt
python -m scripts.sweep_shots --config configs/one_qubit.yaml --checkpoint outputs/one_qubit/best.pt
python -m scripts.sweep_alpha_epsilon --config configs/one_qubit.yaml --checkpoint outputs/one_qubit/best.pt
```

Repeat with the two- and three-qubit configurations.

## Generate all paper plots

After running evaluation and sweeps for the required qubit cases:

```bash
python -m scripts.make_figures --outputs-root outputs --figures-dir figures/results
python -m scripts.make_method_figures --figures-dir figures/method
```

The result plotting script creates:

1. training and validation loss curves;
2. fidelity and trace distance versus replacement fraction `alpha`;
3. fidelity versus frequency-PGD radius;
4. fidelity versus shots per setting;
5. an `alpha`–`epsilon_frequency` robustness heatmap;
6. fidelity distributions by attack;
7. infidelity empirical CDFs;
8. purity versus reconstruction fidelity;
9. clean and attacked performance across qubit counts;
10. true-versus-predicted density-matrix component plots when prediction files are present.

The method-figure script creates:

1. the full reconstruction pipeline;
2. the physical copy-replacement and epsilon-ball relation;
3. the Pauli-cube frequency-vector layout;
4. the modular model interface diagram.

## Run the complete workflow

```bash
python -m scripts.run_all \
  --configs configs/one_qubit.yaml configs/two_qubit.yaml configs/three_qubit.yaml \
  --train --evaluate --sweeps --figures
```

This can be computationally expensive, especially adaptive PGD evaluation for three qubits. Reduce `evaluation.max_samples`, `attack.pgd_eval_steps`, or dataset sizes for a quick test.

## Smoke test

```bash
pytest -q
python -m scripts.train --config configs/smoke_one_qubit.yaml
```
