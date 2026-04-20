from __future__ import annotations

import os
from pathlib import Path

MEDIA_DATA_ROOT_ENV = "DEEPGM_MEDIA_DATA_ROOT"

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / ".data"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"
UPDATED_FEATURE_EXTRACTION_DIR = DATA_DIR / "updated_feature_extraction"
RUNS_DIR = DATA_DIR / "training_runs"
MEDIA_DATA_ROOT = Path(os.getenv(MEDIA_DATA_ROOT_ENV, "/media/Data")).expanduser()
PINMYMETAL_SETS_DIR = MEDIA_DATA_ROOT / "pinmymetal_sets"
MAHOMES_DIR = PINMYMETAL_SETS_DIR / "mahomes"
MAHOMES_TRAIN_SET_DIR = MAHOMES_DIR / "train_set"
MAHOMES_SUMMARY_DIR = MAHOMES_TRAIN_SET_DIR / "data_summarizing_tables"

SUMMARY_TABLE_CSV = MAHOMES_SUMMARY_DIR / "data_summarazing_table.csv"
TRANSITION_METALS_SUMMARY_CSV = MAHOMES_SUMMARY_DIR / "data_summarazing_table_transition_metals.csv"
PREDICTION_RESULTS_SUMMARY_CSV = MAHOMES_SUMMARY_DIR / "prediction_results_summary.csv"
WHETHER_CATALYTIC_SUMMARY_CSV = MAHOMES_SUMMARY_DIR / "data_summarazing_table_transition_metals_whether_catalytic.csv"
CATALYTIC_ONLY_SUMMARY_CSV = MAHOMES_SUMMARY_DIR / "final_data_summarazing_table_transition_metals_only_catalytic.csv"


def get_default_embeddings_dir() -> Path:
    return EMBEDDINGS_DIR


def get_default_updated_feature_extraction_dir() -> Path:
    return UPDATED_FEATURE_EXTRACTION_DIR


def get_default_runs_dir() -> Path:
    return RUNS_DIR


def _resolve_project_dir(
    configured_dir: str | None,
    default_dir: Path,
    create: bool = True,
) -> Path:
    if configured_dir:
        resolved_dir = Path(configured_dir).expanduser()
        if not resolved_dir.is_absolute():
            resolved_dir = PROJECT_ROOT / resolved_dir
    else:
        resolved_dir = default_dir

    if create:
        resolved_dir.mkdir(parents=True, exist_ok=True)
    return resolved_dir


def resolve_embeddings_dir(configured_dir: str | None, create: bool = True) -> Path:
    return _resolve_project_dir(
        configured_dir=configured_dir,
        default_dir=get_default_embeddings_dir(),
        create=create,
    )


def resolve_runs_dir(configured_dir: str | None, create: bool = True) -> Path:
    return _resolve_project_dir(
        configured_dir=configured_dir,
        default_dir=get_default_runs_dir(),
        create=create,
    )
