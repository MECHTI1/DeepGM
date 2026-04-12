from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / ".data"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"
RUNS_DIR = DATA_DIR / "training_runs"
MEDIA_DATA_ROOT = Path("/media/Data")
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


def get_default_runs_dir() -> Path:
    return RUNS_DIR


def resolve_embeddings_dir(configured_dir: str | None, create: bool = True) -> Path:
    if configured_dir:
        embeddings_dir = Path(configured_dir).expanduser()
        if not embeddings_dir.is_absolute():
            embeddings_dir = PROJECT_ROOT / embeddings_dir
    else:
        embeddings_dir = get_default_embeddings_dir()

    if create:
        embeddings_dir.mkdir(parents=True, exist_ok=True)
    return embeddings_dir


def resolve_runs_dir(configured_dir: str | None, create: bool = True) -> Path:
    if configured_dir:
        runs_dir = Path(configured_dir).expanduser()
        if not runs_dir.is_absolute():
            runs_dir = PROJECT_ROOT / runs_dir
    else:
        runs_dir = get_default_runs_dir()

    if create:
        runs_dir.mkdir(parents=True, exist_ok=True)
    return runs_dir
