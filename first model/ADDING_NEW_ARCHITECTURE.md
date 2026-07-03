# Adding another reconstruction architecture

The rest of the repository expects only this interface:

```python
prediction = model(frequencies)
# frequencies: [batch, input_dimension], real
# prediction:  [batch, d, d], complex, physical density matrix
```

The recommended approach is to keep the shared Cholesky conversion in `qst/models/head.py` and replace only the feature extractor.

## Step 1: create the model

Create a file such as `qst/models/transformer.py`. Its final real layer must output `d^2` values and call:

```python
from qst.models.head import raw_cholesky_to_density

rho_hat = raw_cholesky_to_density(raw, dimension)
```

## Step 2: register the model

In `qst/models/registry.py`, import the class, create a builder that accepts `QSTConfig`, and call:

```python
register_model("setting_transformer", build_setting_transformer)
```

## Step 3: change only the YAML configuration

```yaml
model:
  name: setting_transformer
```

No changes are required in state generation, Pauli measurements, attacks, losses, training, evaluation, checkpointing, sweeps, or plotting.
