#!/usr/bin/env python3
from __future__ import annotations

import os
import runpy
from pathlib import Path


os.environ.setdefault("DEEPGM_PINMYMETAL_SET_ROOT", "/media/Data/pinmymetal_sets/test")

SCRIPT_CANDIDATES = (
    "step1b_create_train_structure_files_nonredundant_chains.py",
    "step1b_create_updated_structure_files_nonredundant_chains.py",
)


def resolve_script_path() -> Path:
    base_dir = Path(__file__).resolve().parent
    for candidate in SCRIPT_CANDIDATES:
        candidate_path = base_dir / candidate
        if candidate_path.exists():
            return candidate_path
    raise FileNotFoundError(
        "Could not find a step1b base script. Checked: "
        + ", ".join(str(base_dir / candidate) for candidate in SCRIPT_CANDIDATES)
    )


if __name__ == "__main__":
    runpy.run_path(str(resolve_script_path()), run_name="__main__")
