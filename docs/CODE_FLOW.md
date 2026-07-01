# Execution Flow

## Training command

```bash
python -m scripts.train --config configs/two_qubit.yaml
```

The main calls occur in this order:

1. `load_config(...)` reads and validates the YAML file.
2. `build_dataset(config, "train")` and `build_dataset(config, "validation")` create online or cached datasets.
3. `build_model(config)` uses the model registry to construct `mlp`, `residual_mlp`, or `setting_transformer`.
4. `Trainer.fit(...)` starts the epoch loop.
5. `TomographyDataset.__getitem__(index)` creates each sample:
   - generate a Haar pure state or Ginibre mixed state;
   - generate clean finite-shot Pauli frequencies;
   - choose the configured attack type;
   - create attacked frequencies;
   - return the clean frequencies, attacked frequencies, clean target density, label, and diagnostics.
6. `Trainer._batch_losses(...)` evaluates the clean branch for every sample.
7. The same function evaluates the fixed attacked branch for samples whose label is one.
8. When enabled, `pgd_frequency_attack(...)` creates model-aware adversarial frequencies from a subset of clean vectors.
9. Every model returns `(predicted_density, attack_logit)`.
10. `CholeskyDensityHead` converts `d^2` real outputs into a normalized positive-semidefinite density matrix.
11. The weighted reconstruction, detection, PGD, and measurement-consistency losses are backpropagated.
12. `best.pt`, `last.pt`, and `history.json` are written to the configured output directory.

## Evaluation command

```bash
python -m scripts.evaluate \
  --config configs/two_qubit.yaml \
  --checkpoint outputs/two_qubit/best.pt
```

Evaluation measures clean reconstruction, reconstruction from the fixed attacked inputs, attack classification, metrics for every represented attack type, and a separately generated PGD test.

## Why clean and attacked vectors are both stored

Every generated sample contains:

```text
clean_frequencies  -> measurements from rho
input_frequencies  -> clean or attacked network input
target_density     -> original rho
observed_density   -> effective state after physical replacement, for diagnostics
attack_label       -> 0 or 1
```

The clean branch prevents the model from sacrificing ordinary tomography performance. The attacked branch teaches it to recover the original state and identify the corruption.
