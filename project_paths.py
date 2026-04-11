from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / ".data"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"


def get_default_embeddings_dir() -> Path:
    return EMBEDDINGS_DIR


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
