from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from data_structures import PocketRecord
from training.esm_feature_loading import ResidueKey, load_esm_lookup_for_structure, residue_keys_for_structure_chain
from project_paths import CATALYTIC_ONLY_SUMMARY_CSV, MAHOMES_TRAIN_SET_DIR
from training.feature_sources import (
    build_pocket_feature_coverage,
    resolve_runtime_feature_paths,
)
from training.site_filter import resolve_allowed_site_keys
from training.structure_loading import (
    build_load_report,
    find_structure_files,
    load_structure_pockets,
    pocket_has_full_supervision,
)


DEFAULT_STRUCTURE_DIR = MAHOMES_TRAIN_SET_DIR
DEFAULT_TRAIN_SUMMARY_CSV = CATALYTIC_ONLY_SUMMARY_CSV


@dataclass(frozen=True)
class PocketLoadResult:
    pockets: List[PocketRecord]
    feature_report: Dict[str, Any]


def _assemble_pocket_load_result(
    *,
    pockets: List[PocketRecord],
    structure_files: List[Path],
    skipped_structures: List[Dict[str, str]],
    feature_fallbacks: List[Dict[str, str]],
) -> PocketLoadResult:
    return PocketLoadResult(
        pockets=pockets,
        feature_report=build_load_report(
            pockets=pockets,
            structure_files=structure_files,
            skipped_structures=skipped_structures,
            feature_fallbacks=feature_fallbacks,
        ),
    )


def load_labeled_pockets_with_report_from_dir(
    structure_dir: Path,
    max_cases: Optional[int] = None,
    require_full_labels: bool = True,
    summary_csv: Optional[Path] = DEFAULT_TRAIN_SUMMARY_CSV,
    esm_dim: int = 256,
    esm_embeddings_dir: str | Path | None = None,
    require_esm_embeddings: bool = True,
    external_features_root_dir: str | Path | None = None,
    require_external_features: bool = True,
) -> PocketLoadResult:
    structure_root = Path(structure_dir)
    structure_files = find_structure_files(structure_root)
    if not structure_files:
        raise FileNotFoundError(f"No structure files found under {structure_root}")

    allowed_site_keys = resolve_allowed_site_keys(summary_csv)
    embeddings_dir, feature_root_dir = resolve_runtime_feature_paths(
        structure_dir=structure_root,
        esm_embeddings_dir=esm_embeddings_dir,
        external_features_root_dir=external_features_root_dir,
    )

    pockets: List[PocketRecord] = []
    skipped_structures: List[Dict[str, str]] = []
    feature_fallbacks: List[Dict[str, str]] = []

    for structure_path in structure_files:
        structure_pockets, structure_fallbacks = load_structure_pockets(
            structure_path=structure_path,
            structure_root=structure_root,
            allowed_site_keys=allowed_site_keys,
            esm_dim=esm_dim,
            embeddings_dir=embeddings_dir,
            require_esm_embeddings=require_esm_embeddings,
            feature_root_dir=feature_root_dir,
            require_external_features=require_external_features,
        )
        feature_fallbacks.extend(structure_fallbacks)

        for pocket in structure_pockets:
            if require_full_labels and not pocket_has_full_supervision(pocket):
                continue
            pockets.append(pocket)
            if max_cases is not None and len(pockets) >= max_cases:
                return _assemble_pocket_load_result(
                    pockets=pockets,
                    structure_files=structure_files,
                    skipped_structures=skipped_structures,
                    feature_fallbacks=feature_fallbacks,
                )

    if not pockets:
        if require_full_labels:
            raise ValueError(f"No fully labeled metal-centered pockets were extracted from {structure_root}")
        raise ValueError(f"No metal-centered pockets were extracted from {structure_root}")

    return _assemble_pocket_load_result(
        pockets=pockets,
        structure_files=structure_files,
        skipped_structures=skipped_structures,
        feature_fallbacks=feature_fallbacks,
    )


def load_labeled_pockets_from_dir(
    structure_dir: Path,
    max_cases: Optional[int] = None,
    require_full_labels: bool = True,
    summary_csv: Optional[Path] = DEFAULT_TRAIN_SUMMARY_CSV,
    esm_dim: int = 256,
    esm_embeddings_dir: str | Path | None = None,
    require_esm_embeddings: bool = True,
    external_features_root_dir: str | Path | None = None,
    require_external_features: bool = True,
) -> List[PocketRecord]:
    return load_labeled_pockets_with_report_from_dir(
        structure_dir=structure_dir,
        max_cases=max_cases,
        require_full_labels=require_full_labels,
        summary_csv=summary_csv,
        esm_dim=esm_dim,
        esm_embeddings_dir=esm_embeddings_dir,
        require_esm_embeddings=require_esm_embeddings,
        external_features_root_dir=external_features_root_dir,
        require_external_features=require_external_features,
    ).pockets


def load_training_pockets_from_dir(
    structure_dir: Path,
    require_full_labels: bool = True,
    summary_csv: Optional[Path] = DEFAULT_TRAIN_SUMMARY_CSV,
    esm_dim: int = 256,
    esm_embeddings_dir: str | Path | None = None,
    require_esm_embeddings: bool = True,
    external_features_root_dir: str | Path | None = None,
    require_external_features: bool = True,
) -> List[PocketRecord]:
    return load_labeled_pockets_from_dir(
        structure_dir=structure_dir,
        max_cases=None,
        require_full_labels=require_full_labels,
        summary_csv=summary_csv,
        esm_dim=esm_dim,
        esm_embeddings_dir=esm_embeddings_dir,
        require_esm_embeddings=require_esm_embeddings,
        external_features_root_dir=external_features_root_dir,
        require_external_features=require_external_features,
    )


def load_training_pockets_with_report_from_dir(
    structure_dir: Path,
    require_full_labels: bool = True,
    summary_csv: Optional[Path] = DEFAULT_TRAIN_SUMMARY_CSV,
    esm_dim: int = 256,
    esm_embeddings_dir: str | Path | None = None,
    require_esm_embeddings: bool = True,
    external_features_root_dir: str | Path | None = None,
    require_external_features: bool = True,
) -> PocketLoadResult:
    return load_labeled_pockets_with_report_from_dir(
        structure_dir=structure_dir,
        max_cases=None,
        require_full_labels=require_full_labels,
        summary_csv=summary_csv,
        esm_dim=esm_dim,
        esm_embeddings_dir=esm_embeddings_dir,
        require_esm_embeddings=require_esm_embeddings,
        external_features_root_dir=external_features_root_dir,
        require_external_features=require_external_features,
    )


def load_smoke_test_pockets_from_dir(
    structure_dir: Path,
    max_cases: int = 4,
    require_full_labels: bool = True,
    summary_csv: Optional[Path] = DEFAULT_TRAIN_SUMMARY_CSV,
    esm_dim: int = 256,
    esm_embeddings_dir: str | Path | None = None,
    require_esm_embeddings: bool = False,
    external_features_root_dir: str | Path | None = None,
    require_external_features: bool = False,
) -> List[PocketRecord]:
    return load_labeled_pockets_from_dir(
        structure_dir=structure_dir,
        max_cases=max_cases,
        require_full_labels=require_full_labels,
        summary_csv=summary_csv,
        esm_dim=esm_dim,
        esm_embeddings_dir=esm_embeddings_dir,
        require_esm_embeddings=require_esm_embeddings,
        external_features_root_dir=external_features_root_dir,
        require_external_features=require_external_features,
    )


__all__ = [
    "DEFAULT_STRUCTURE_DIR",
    "DEFAULT_TRAIN_SUMMARY_CSV",
    "PocketLoadResult",
    "ResidueKey",
    "build_pocket_feature_coverage",
    "find_structure_files",
    "load_esm_lookup_for_structure",
    "load_labeled_pockets_from_dir",
    "load_labeled_pockets_with_report_from_dir",
    "load_smoke_test_pockets_from_dir",
    "load_training_pockets_from_dir",
    "load_training_pockets_with_report_from_dir",
    "residue_keys_for_structure_chain",
]
