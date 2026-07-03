from __future__ import annotations

import argparse

from qst.trainer import RobustQSTTrainer
from scripts.common import prepare


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    arguments = parser.parse_args()
    config, device, _, loaders, measurement, model = prepare(arguments.config)
    trainer = RobustQSTTrainer(config, model, measurement, device)
    summary = trainer.fit(loaders["train"], loaders["val"])
    print(summary)


if __name__ == "__main__":
    main()
