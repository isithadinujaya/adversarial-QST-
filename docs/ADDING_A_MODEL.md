# Adding or Replacing the Neural Network

The quantum simulation, attacks, dataset, losses, trainer, and evaluator do not depend on a particular neural architecture.

Create a new file such as `qst/models/my_model.py`:

```python
import torch
from torch import nn

from qst.config import ExperimentConfig
from qst.models.base import QSTModelBase
from qst.models.registry import register_model


@register_model("my_model")
class MyModel(QSTModelBase):
    def __init__(self, experiment_config: ExperimentConfig) -> None:
        feature_dimension = 256
        super().__init__(experiment_config, feature_dimension)
        self.feature_network = nn.Sequential(
            nn.Linear(experiment_config.quantum.input_dimension, feature_dimension),
            nn.GELU(),
        )

    def extract_features(self, frequencies: torch.Tensor) -> torch.Tensor:
        return self.feature_network(frequencies)
```

Import the new module in `qst/models/__init__.py` so its registration decorator runs, then select it in YAML:

```yaml
model:
  name: my_model
```

`QSTModelBase.forward(...)` automatically applies the shared physical density head and binary detection head. A future architecture therefore only needs to implement `extract_features(...)`.

A completely custom forward method is also possible, but it must preserve this public interface:

```python
predicted_density, attack_logit = model(frequencies)
```

where `predicted_density` has shape `[batch, d, d]` and `attack_logit` has shape `[batch]`.
