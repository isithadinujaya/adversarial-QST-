from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from qst.config import load_config


def run(command: list[str]) -> None:
    print("+", " ".join(command))
    subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+", required=True)
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--sweeps", action="store_true")
    parser.add_argument("--figures", action="store_true")
    arguments = parser.parse_args()

    for config_path in arguments.configs:
        config = load_config(config_path)
        checkpoint = str(Path(config.experiment.output_dir) / "best.pt")
        if arguments.train:
            run([sys.executable, "-m", "scripts.train", "--config", config_path])
        if arguments.evaluate:
            run([sys.executable, "-m", "scripts.evaluate", "--config", config_path, "--checkpoint", checkpoint])
        if arguments.sweeps:
            for module in [
                "scripts.sweep_alpha",
                "scripts.sweep_epsilon",
                "scripts.sweep_shots",
                "scripts.sweep_alpha_epsilon",
            ]:
                run([sys.executable, "-m", module, "--config", config_path, "--checkpoint", checkpoint])

    if arguments.figures:
        run([sys.executable, "-m", "scripts.make_figures"])
        run([sys.executable, "-m", "scripts.make_method_figures"])


if __name__ == "__main__":
    main()
