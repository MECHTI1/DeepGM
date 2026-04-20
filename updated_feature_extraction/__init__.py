"""Modern residue-level external feature generation for DeepGM.

This package replaces the legacy Bluues/Rosetta-derived feature pipeline
with a lighter workflow built around Biotite and PROPKA while keeping the
same feature names expected by the training code.
"""

from .constants import FEATURE_NAMES
from .core import build_structure_feature_payload, generate_feature_map_for_structure

__all__ = [
    "FEATURE_NAMES",
    "build_structure_feature_payload",
    "generate_feature_map_for_structure",
]
