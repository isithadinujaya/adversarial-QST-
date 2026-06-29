# Validation performed

The project was checked in a CPU environment with Python 3.13 and PyTorch 2.10.

## Automated tests

Command:

```bash
pytest -q
```

Result:

```text
8 passed
```

The tests cover state physicality, Pauli sampling, both random-attack simulation modes, distant target selection, targeted attack validity, physical neural-network outputs, PGD constraints, and a complete backward pass.

## End-to-end training smoke test

Command:

```bash
python train.py --debug --device cpu
```

The one-epoch debug run completed and saved a checkpoint. This confirms that state generation, measurements, all three training attacks, both network heads, the combined loss, optimization and checkpoint saving work together.

## Evaluation smoke test

A small checkpoint-loading evaluation completed for clean data, all physical contamination strengths, standard PGD and adaptive PGD. The smoke run used only six states, two PGD steps and one restart, so its numerical results are not research results; it only verifies control flow and CSV creation.
