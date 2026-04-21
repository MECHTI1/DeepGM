from __future__ import annotations

from pathlib import Path

from project_paths import get_default_updated_feature_extraction_dir, resolve_embeddings_dir

VALID_EXTERNAL_FEATURE_SOURCE_CHOICES = ("auto", "bluues_rosetta", "updated")


def resolve_external_feature_root_dir(
    *,
    structure_dir: Path,
    external_features_root_dir: str | Path | None,
    external_feature_source: str,
) -> Path:
    if external_feature_source not in VALID_EXTERNAL_FEATURE_SOURCE_CHOICES:
        raise ValueError(
            f"Unsupported external feature source: {external_feature_source!r}. "
            f"Expected one of: {', '.join(repr(choice) for choice in VALID_EXTERNAL_FEATURE_SOURCE_CHOICES)}."
        )

    if external_features_root_dir is not None:
        return Path(external_features_root_dir)

    if external_feature_source == "bluues_rosetta":
        return structure_dir
    if external_feature_source == "updated":
        return get_default_updated_feature_extraction_dir()

    default_updated_feature_dir = get_default_updated_feature_extraction_dir()
    return default_updated_feature_dir if default_updated_feature_dir.exists() else structure_dir


def resolve_runtime_feature_paths(
    *,
    structure_dir: Path,
    esm_embeddings_dir: str | Path | None,
    external_features_root_dir: str | Path | None,
    external_feature_source: str = "auto",
) -> tuple[Path, Path]:
    embeddings_dir = (
        resolve_embeddings_dir(str(esm_embeddings_dir), create=False)
        if esm_embeddings_dir is not None
        else resolve_embeddings_dir(None, create=False)
    )
    feature_root_dir = resolve_external_feature_root_dir(
        structure_dir=structure_dir,
        external_features_root_dir=external_features_root_dir,
        external_feature_source=external_feature_source,
    )
    return embeddings_dir, feature_root_dir
