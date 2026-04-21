from __future__ import annotations

from typing import Sequence

from training.run import run_training
from training.task_entrypoint import parse_separate_task_args


def main(argv: Sequence[str] | None = None) -> None:
    run_training(parse_separate_task_args("ec", argv))


if __name__ == "__main__":
    main()
