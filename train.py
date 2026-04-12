from __future__ import annotations

from training.config import parse_args
from training.run import run_training


def main() -> None:
    run_training(parse_args())


if __name__ == "__main__":
    main()
