"""Shared training defaults.

This module centralizes default dataset paths used by training configuration
and data-loading entry points.
"""

from __future__ import annotations

from project_paths import CATALYTIC_ONLY_SUMMARY_CSV, MAHOMES_TRAIN_SET_DIR

DEFAULT_STRUCTURE_DIR = MAHOMES_TRAIN_SET_DIR
DEFAULT_TRAIN_SUMMARY_CSV = CATALYTIC_ONLY_SUMMARY_CSV
