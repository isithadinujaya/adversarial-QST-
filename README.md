# Adversarial Quantum State Tomography (1–3 Qubits)

This project implements the complete pipeline discussed for robust neural-network quantum state tomography:

\[
\rho \longrightarrow \text{finite-shot Pauli measurements}
\longrightarrow \mathbf f
\longrightarrow \text{clean or adversarial corruption}
\longrightarrow \text{neural network}
\longrightarrow (\hat\rho,\;\text{attack logit}).
\]

The same codebase supports one, two, and three qubits. Each qubit case uses a separate configuration, dataset realization, and model checkpoint because the input and output dimensions change.

## Implemented design

- Haar-random pure states.
- Ginibre-ensemble mixed states.
- Configurable pure/mixed state ratio.
- Full local Pauli measurement settings from \(\{X,Y,Z\}^{\otimes n}\).
- Practical finite-shot frequencies using multinomial sampling.
- Default: 1000 copies for every measurement setting.
- Exact copy-replacement simulation: for attack fraction \(\epsilon\), each setting uses \(N-K\) copies of the clean state and \(K\) copies of the replacement state, where \(K=\operatorname{round}(\epsilon N)\).
- Random replacement attack with pure and mixed replacement states.
- Targeted replacement attack toward a configurable fixed target state.
- Worst-case replacement attack selected from random candidates by maximum trace distance.
- Random frequency-vector attack constrained separately inside every measurement-setting probability simplex.
- Model-aware PGD frequency attack generated online during training.
- A reconstruction head that always outputs a physical density matrix using
  \[
  \hat\rho=TT^\dagger/\operatorname{Tr}(TT^\dagger).
  \]
- A binary attack-detection head.
- Clean reconstruction loss, adversarial reconstruction loss, attack-detection loss, and optional clean measurement-consistency loss.
- Model registry so the neural architecture can be changed without rewriting the quantum, attack, data, or training code.
- Included models: `mlp`, `residual_mlp`, and `setting_transformer`.

## Dimensions

| Qubits | Hilbert dimension | Settings | Outcomes/setting | Frequency length | Cholesky outputs |
|---:|---:|---:|---:|---:|---:|
| 1 | 2 | 3 | 2 | 6 | 4 |
| 2 | 4 | 9 | 4 | 36 | 16 |
| 3 | 8 | 27 | 8 | 216 | 64 |

The frequency vector is flattened in setting-major, outcome-major order. For two qubits, settings are ordered as
`XX, XY, XZ, YX, YY, YZ, ZX, ZY, ZZ`, and every setting contains outcomes `00, 01, 10, 11`.

## Project structure

```text
adversarial_qst_multiqubit/
├── configs/                   # one-, two-, and three-qubit experiments
├── qst/
│   ├── attacks/               # physical, frequency, and PGD attacks
│   ├── data/                  # online and cached datasets
│   ├── models/                # model registry and architectures
│   ├── quantum/               # states, measurements, and metrics
│   ├── training/              # losses, trainer, checkpoints
│   └── config.py
├── scripts/
│   ├── train.py
│   ├── evaluate.py
│   ├── generate_dataset.py
│   ├── inspect_sample.py
│   └── smoke_test.py
└── tests/
```

## Installation in Windows Git Bash

From the project folder:

```bash
python -m venv .venv
source .venv/Scripts/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

On Linux or Raspberry Pi, activate with:

```bash
source .venv/bin/activate
```

## Verify the installation

```bash
python -m scripts.smoke_test
pytest -q
```

## Train

One qubit:

```bash
python -m scripts.train --config configs/one_qubit.yaml
```

Two qubits:

```bash
python -m scripts.train --config configs/two_qubit.yaml
```

Three qubits:

```bash
python -m scripts.train --config configs/three_qubit.yaml
```

The default device is `auto`, so CUDA is used when available. Each configuration saves into a separate output directory.

## Evaluate a checkpoint

```bash
python -m scripts.evaluate \
  --config configs/two_qubit.yaml \
  --checkpoint outputs/two_qubit/best.pt
```

The evaluation report includes clean and attacked reconstruction errors, trace distance, fidelity, attack accuracy, precision, recall, F1, AUROC, per-attack reconstruction metrics, and a separately generated model-aware PGD evaluation.

## Inspect one generated sample

```bash
python -m scripts.inspect_sample --config configs/two_qubit.yaml --index 0
```

## Optional cached datasets

The default mode generates samples deterministically on demand. To create fixed files:

```bash
python -m scripts.generate_dataset \
  --config configs/two_qubit.yaml \
  --split train \
  --output data/two_qubit_train.pt
```

Then set `data.mode: cached` and the corresponding file paths in the YAML configuration.

## Changing the neural network

Only change this field:

```yaml
model:
  name: residual_mlp
```

Available names:

```text
mlp
residual_mlp
setting_transformer
```

All models receive the same frequency vector and return:

```python
predicted_density, attack_logit = model(frequencies)
```

### Adding a future model

Create a file under `qst/models/`, register the class, and keep the same forward interface:

```python
from qst.models.base import QSTModelBase
from qst.models.registry import register_model

@register_model("my_future_model")
class MyFutureModel(QSTModelBase):
    def __init__(self, experiment_config):
        super().__init__(experiment_config)
        # Build feature extractor and heads.

    def forward(self, frequencies):
        features = ...
        density = self.density_head(features)
        attack_logit = self.detection_head(features).squeeze(-1)
        return density, attack_logit
```

No state generation, measurement, attack, dataset, loss, training, or evaluation code needs to be changed.

## Important interpretation of the training target

For an attacked sample, the input frequencies come from corrupted physical copies or from a corrupted frequency vector, but the reconstruction target remains the original clean density matrix \(\rho\). Therefore, the model is explicitly trained to:

1. estimate the state that should have been measured before the attack, and
2. report that the input was attacked.

The returned `observed_density` is stored only for diagnostics. For a replacement attack it equals

\[
\rho_{\mathrm{observed}}=(1-\epsilon)\rho+\epsilon\sigma,
\]

while the counts are generated from the exact split of clean and replacement copies.
