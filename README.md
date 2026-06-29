# Adversarially Robust Neural-Network Quantum State Tomography

This project implements the agreed single-qubit QST pipeline:

- Haar-random pure states
- Ginibre mixed states
- depolarized pure states
- Pauli X, Y and Z measurements
- **1000 copies for each basis** (3000 physical copies per state)
- random physical copy replacement
- targeted physical copy replacement
- standard and adaptive L-infinity PGD frequency attacks
- Cholesky-parameterized density-matrix reconstruction
- binary attack detection

## 1. Threat models

### Random physical replacement

For every basis, `floor(alpha * 1000)` copies are replaced. Every corrupted copy conceptually receives an independently sampled replacement state. Each replacement is Haar-pure with probability 0.5 and Ginibre-mixed with probability 0.5.

Because both ensembles are unitarily invariant and average to the maximally mixed state, an independently redrawn replacement gives a `+` result with probability exactly 0.5 in every Pauli basis after marginalizing over the random state. Training therefore uses an equivalent and much faster binomial sampler. The attack function also provides `simulation_mode="explicit"` to generate every replacement density matrix for small verification experiments.

This differs from a targeted attack because the replacements are not selected to move the estimate toward one chosen state.

### Targeted physical replacement

One pure/mixed target state is chosen for each clean state, subject to

`trace_distance(clean, target) >= 0.5`.

Every corrupted copy in all three basis batches is replaced by that same target state.

### PGD frequency attack

PGD perturbs the classical six-dimensional input

`[f_X+, f_X-, f_Y+, f_Y-, f_Z+, f_Z-]`.

Only the three positive frequencies are optimized. The negative frequencies are reconstructed as complements, so each basis always satisfies

`f_B+ + f_B- = 1`.

The code implements:

- `standard`: maximize density-matrix reconstruction loss
- `adaptive`: maximize reconstruction loss while encouraging the detector to predict clean

## 2. Installation

Create a virtual environment and install the requirements:

```bash
python -m venv .venv
```

Linux/macOS:

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

For a CUDA-specific PyTorch installation, use the command provided by the official PyTorch installation selector rather than relying on the generic requirement line.

## 3. Verify the implementation

```bash
pytest -q
```

The tests verify:

- generated density matrices are Hermitian, positive semidefinite and trace one
- Pauli frequency pairs are valid
- physical attacks return valid frequencies
- targeted states satisfy the distance requirement
- PGD respects its L-infinity budget
- the network output is a physical density matrix
- a complete backward pass succeeds

## 4. Small end-to-end run

Run this first:

```bash
python train.py --debug
```

The debug mode uses 384 training states, 96 validation states, one epoch and two PGD steps. Its purpose is to verify the complete pipeline, not to produce final research results.

## 5. Full training

```bash
python train.py --device auto
```

Defaults:

- 50,000 training states
- 5,000 validation states
- 10,000 test states
- batch size 256
- 50 epochs
- 10-step standard PGD during training
- equal numbers of random physical, targeted physical and PGD examples in each batch

The best checkpoint is selected using validation adversarial infidelity.

Useful overrides:

```bash
python train.py --epochs 100 --batch-size 128 --checkpoint checkpoints/run_01.pt
```

## 6. Evaluation

A quick evaluation:

```bash
python evaluate.py checkpoints/best_model.pt --max-states 100 --pgd-restarts 1 --pgd-steps 5
```

Final evaluation:

```bash
python evaluate.py checkpoints/best_model.pt
```

The full evaluation tests:

- clean data
- random physical attack at alpha = 0.01 to 0.30
- targeted physical attack at alpha = 0.01 to 0.30
- standard PGD at epsilon = 0.005 to 0.05
- adaptive PGD at epsilon = 0.005 to 0.05
- 40 PGD steps and five random restarts by default

Results are written to `results/evaluation.csv`.

## 7. Loss

The training objective is

`L = L_clean + L_adversarial + 0.1 L_detection`.

Each reconstruction loss is

`||rho_hat-rho||_F^2 + 0.1(1-F(rho_hat,rho))`.

The detection loss uses logits with binary cross entropy. Clean inputs have label zero and every attack type has label one.

## 8. Important interpretation

The model is trained to reconstruct the original clean state from an attacked frequency vector. For a targeted attack, the training label remains the original state—not the target state.

The physical contamination fraction `alpha` and frequency-space PGD radius `epsilon` are different quantities and should be reported separately.

## 9. Detection limitation

The detector should be interpreted as an **attack-likelihood estimator under the training distribution**, not as a mathematical certificate. A physical replacement batch produces measurement statistics of an effective valid density matrix. From a single frequency vector alone, that effective state can be indistinguishable from a legitimate uncorrupted state with the same density matrix.

Therefore, the primary scientifically defensible objective is robust reconstruction. Detection performance is still worth measuring, especially for PGD vectors that may be pair-normalized but inconsistent with a physical Bloch vector, but perfect detection of physical replacement attacks is not generally identifiable from this input alone.

## 10. Runtime note

PGD adversarial training is compute-intensive. The full configuration is intended for a GPU. Start with `--debug`, then increase dataset size, epochs, PGD steps and evaluation restarts gradually.
