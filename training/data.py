"""Load training pockets from structure files.

This module:
- discovers structure files under a dataset directory, such as `.pdb` and `.cif`
- resolves the locations of runtime feature inputs, including ESM embeddings and external features
- filters out pockets that should not be used, such as pockets excluded by the summary CSV or missing required supervision
- returns either the loaded pockets alone or the loaded pockets together with a feature-coverage report
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from data_structures import PocketRecord
from training.defaults import DEFAULT_STRUCTURE_DIR, DEFAULT_TRAIN_SUMMARY_CSV
from training.esm_feature_loading import DEFAULT_ESMC_EMBED_DIM
from training.feature_paths import resolve_runtime_feature_paths
from training.site_filter import resolve_allowed_site_metal_labels
from training.structure_loading import (
    StructureLoadError,
    build_load_report,
    find_structure_files,
    load_structure_pockets,
    pocket_has_required_supervision,
)


@dataclass(frozen=True)
class PocketLoadResult:
    pockets: List[PocketRecord]
    feature_report: Dict[str, Any]


def _assemble_pocket_load_result(
    *,
    pockets: List[PocketRecord],
    structure_files: List[Path],
    feature_fallbacks: List[Dict[str, str]],
    skipped_pockets: List[Dict[str, str]],
    invalid_structures: List[Dict[str, str]],
) -> PocketLoadResult:
    """Package loaded pockets together with a summarized feature-load report."""
    return PocketLoadResult(
        pockets=pockets,
        feature_report=build_load_report(
            pockets=pockets,
            structure_files=structure_files,
            feature_fallbacks=feature_fallbacks,
            skipped_pockets=skipped_pockets,
            invalid_structures=invalid_structures,
        ),
    )


def load_labeled_pockets_with_report_from_dir(
    structure_dir: Path,
    max_cases: Optional[int] = None,
    require_full_labels: bool = True,
    required_targets: tuple[str, ...] = ("metal", "ec"),
    summary_csv: Optional[Path] = DEFAULT_TRAIN_SUMMARY_CSV,
    esm_dim: int = DEFAULT_ESMC_EMBED_DIM,
    esm_embeddings_dir: str | Path | None = None,
    require_esm_embeddings: bool = True,
    external_features_root_dir: str | Path | None = None,
    external_feature_source: str = "auto",
    require_external_features: bool = True,
    unsupported_metal_policy: str = "error",
    invalid_structure_policy: str = "skip",
) -> PocketLoadResult:
    """Load labeled pockets from a structure directory and return them with a load report."""
    structure_root = Path(structure_dir)
    structure_files = find_structure_files(structure_root)
    if not structure_files:
        raise FileNotFoundError(f"No structure files found under {structure_root}")

    allowed_site_metal_labels = resolve_allowed_site_metal_labels(summary_csv)
    embeddings_dir, feature_root_dir = resolve_runtime_feature_paths(
        structure_dir=structure_root,
        esm_embeddings_dir=esm_embeddings_dir,
        external_features_root_dir=external_features_root_dir,
        external_feature_source=external_feature_source,
    )

    pockets: List[PocketRecord] = []
    feature_fallbacks: List[Dict[str, str]] = []
    skipped_pockets: List[Dict[str, str]] = []
    invalid_structures: List[Dict[str, str]] = []

    for structure_path in structure_files:
        try:
            structure_pockets, structure_fallbacks, structure_skipped_pockets = load_structure_pockets(
                structure_path=structure_path,
                structure_root=structure_root,
                allowed_site_metal_labels=allowed_site_metal_labels,
                esm_dim=esm_dim,
                embeddings_dir=embeddings_dir,
                require_esm_embeddings=require_esm_embeddings,
                feature_root_dir=feature_root_dir,
                external_feature_source=external_feature_source,
                require_external_features=require_external_features,
                unsupported_metal_policy=unsupported_metal_policy,
            )
        except StructureLoadError as exc:
            if invalid_structure_policy != "skip":
                raise
            invalid_structures.append(
                {
                    "structure_path": str(structure_path),
                    "reason": "invalid_structure",
                    "detail": str(exc),
                }
            )
            continue
        feature_fallbacks.extend(structure_fallbacks)
        skipped_pockets.extend(structure_skipped_pockets)

        for pocket in structure_pockets:
            if require_full_labels and not pocket_has_required_supervision(
                pocket,
                required_targets=required_targets,
            ):
                skipped_pockets.append(
                    {
                        "structure_id": pocket.structure_id,
                        "pocket_id": pocket.pocket_id,
                        "reason": "missing_required_supervision",
                    }
                )
                continue
            pockets.append(pocket)
            if max_cases is not None and len(pockets) >= max_cases:
                return _assemble_pocket_load_result(
                    pockets=pockets,
                    structure_files=structure_files,
                    feature_fallbacks=feature_fallbacks,
                    skipped_pockets=skipped_pockets,
                    invalid_structures=invalid_structures,
                )

    if not pockets:
        if require_full_labels:
            required_target_list = ", ".join(required_targets)
            raise ValueError(
                "No metal-centered pockets with required supervision "
                f"({required_target_list}) were extracted from {structure_root}"
            )
        raise ValueError(f"No metal-centered pockets were extracted from {structure_root}")

    return _assemble_pocket_load_result(
        pockets=pockets,
        structure_files=structure_files,
        feature_fallbacks=feature_fallbacks,
        skipped_pockets=skipped_pockets,
        invalid_structures=invalid_structures,
    )


def load_training_pockets_with_report_from_dir(
    structure_dir: Path,
    require_full_labels: bool = True,
    required_targets: tuple[str, ...] = ("metal", "ec"),
    summary_csv: Optional[Path] = DEFAULT_TRAIN_SUMMARY_CSV,
    esm_dim: int = DEFAULT_ESMC_EMBED_DIM,
    esm_embeddings_dir: str | Path | None = None,
    require_esm_embeddings: bool = True,
    external_features_root_dir: str | Path | None = None,
    external_feature_source: str = "auto",
    require_external_features: bool = True,
    unsupported_metal_policy: str = "error",
    invalid_structure_policy: str = "skip",
) -> PocketLoadResult:
    """Load the full training set from a structure directory with a load report."""
    return load_labeled_pockets_with_report_from_dir(
        structure_dir=structure_dir,
        max_cases=None,
        require_full_labels=require_full_labels,
        required_targets=required_targets,
        summary_csv=summary_csv,
        esm_dim=esm_dim,
        esm_embeddings_dir=esm_embeddings_dir,
        require_esm_embeddings=require_esm_embeddings,
        external_features_root_dir=external_features_root_dir,
        external_feature_source=external_feature_source,
        require_external_features=require_external_features,
        unsupported_metal_policy=unsupported_metal_policy,
        invalid_structure_policy=invalid_structure_policy,
    )


def load_smoke_test_pockets_from_dir(
    structure_dir: Path,
    max_cases: int = 4,
    require_full_labels: bool = True,
    required_targets: tuple[str, ...] = ("metal", "ec"),
    summary_csv: Optional[Path] = DEFAULT_TRAIN_SUMMARY_CSV,
    esm_dim: int = DEFAULT_ESMC_EMBED_DIM,
    esm_embeddings_dir: str | Path | None = None,
    require_esm_embeddings: bool = False,
    external_features_root_dir: str | Path | None = None,
    external_feature_source: str = "auto",
    require_external_features: bool = False,
    unsupported_metal_policy: str = "error",
    invalid_structure_policy: str = "skip",
) -> List[PocketRecord]:
    """Load a small pocket subset for smoke tests, with optional feature requirements relaxed."""
    return load_labeled_pockets_with_report_from_dir(
        structure_dir=structure_dir,
        max_cases=max_cases,
        require_full_labels=require_full_labels,
        required_targets=required_targets,
        summary_csv=summary_csv,
        esm_dim=esm_dim,
        esm_embeddings_dir=esm_embeddings_dir,
        require_esm_embeddings=require_esm_embeddings,
        external_features_root_dir=external_features_root_dir,
        external_feature_source=external_feature_source,
        require_external_features=require_external_features,
        unsupported_metal_policy=unsupported_metal_policy,
        invalid_structure_policy=invalid_structure_policy,
    ).pockets
