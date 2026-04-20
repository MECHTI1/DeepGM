from __future__ import annotations

from pathlib import Path

from project_paths import get_default_updated_feature_extraction_dir, resolve_embeddings_dir


def resolve_runtime_feature_paths(
    *,
    structure_dir: Path,
    esm_embeddings_dir: str | Path | None,
    external_features_root_dir: str | Path | None,
) -> tuple[Path, Path]:
    embeddings_dir = (
        resolve_embeddings_dir(str(esm_embeddings_dir), create=False)
        if esm_embeddings_dir is not None
        else resolve_embeddings_dir(None, create=False)
    )
    if external_features_root_dir is not None:
        feature_root_dir = Path(external_features_root_dir)
    else:
        default_updated_feature_dir = get_default_updated_feature_extraction_dir()
        feature_root_dir = default_updated_feature_dir if default_updated_feature_dir.exists() else structure_dir
    return embeddings_dir, feature_root_dir
