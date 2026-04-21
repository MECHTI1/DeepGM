from __future__ import annotations

import sys
from typing import Sequence

from training.run import run_training
from training.task_entrypoint import parse_separate_task_args


def main(argv: Sequence[str] | None = None) -> None:
    effective_argv = list(sys.argv[1:] if argv is None else argv)
    run_training(parse_separate_task_args("metal", effective_argv))


if __name__ == "__main__":
    main()
