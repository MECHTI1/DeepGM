#!/usr/bin/env python3

"""
Compatibility facade for the DeepGM pocket-graph prototype.

The implementation has been split into focused modules:
- data_structures.py
- featurization.py
- graph_construction.py
- model.py
- train_utils.py

This file re-exports the existing public API so prior `import main` usage
continues to work.
"""

from data_structures import *  # noqa: F401,F403
from featurization import *  # noqa: F401,F403
from graph_construction import *  # noqa: F401,F403
from model import *  # noqa: F401,F403
from train_utils import *  # noqa: F401,F403


if __name__ == "__main__":
    run_smoke_test(device="cpu")

