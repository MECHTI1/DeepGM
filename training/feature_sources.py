from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from torch import Tensor

from data_structures import PocketRecord
from graph.feature_utils import attach_esm_embeddings, attach_external_residue_features
from project_paths import resolve_embeddings_dir
from temp_helper.return_relevant_features_val import structure_dir_to_feature_lookup
from training.esm_feature_loading import ResidueKey, load_esm_lookup_for_structure


@dataclass(frozen=True)
class StructureFeatureSources:
    esm_lookup: Dict[ResidueKey, Tensor]
    external_feature_lookup: Dict[ResidueKey, Dict[str, float]]


def feature_fallback_record(
    structure_path: Path,
    *,
    feature_name: str,
    detail: str,
) -> Dict[str, str]:
    return {
        "structure_path": str(structure_path),
        "feature": feature_name,
        "detail": detail,
    }


def resolve_runtime_feature_paths(
    *,
    structure_dir: Path,
    esm_embeddings_dir: str | Path | None,
    external_features_root_dir: str | Path | None,
) -> Tuple[Path, Path]:
    embeddings_dir = (
        resolve_embeddings_dir(str(esm_embeddings_dir), create=False)
        if esm_embeddings_dir is not None
        else resolve_embeddings_dir(None, create=False)
    )
    feature_root_dir = Path(external_features_root_dir) if external_features_root_dir is not None else structure_dir
    return embeddings_dir, feature_root_dir


def resolve_structure_feature_dir(
    *,
    structure_path: Path,
    structure_root: Path,
    feature_root_dir: Optional[Path],
) -> Optional[Path]:
    direct_candidate = structure_path.parent / structure_path.stem
    if direct_candidate.is_dir():
        return direct_candidate
    if feature_root_dir is None:
        return None

    feature_root_dir = Path(feature_root_dir)
    try:
        relative_parent = structure_path.parent.relative_to(structure_root)
    except ValueError:
        relative_parent = Path()

    for candidate in (
        feature_root_dir / relative_parent / structure_path.stem,
        feature_root_dir / structure_path.stem,
    ):
        if candidate.is_dir():
            return candidate
    return None


def load_external_feature_lookup_for_structure(
    *,
    structure_path: Path,
    structure_root: Path,
    feature_root_dir: Optional[Path],
) -> Dict[ResidueKey, Dict[str, float]]:
    feature_dir = resolve_structure_feature_dir(
        structure_path=structure_path,
        structure_root=structure_root,
        feature_root_dir=feature_root_dir,
    )
    if feature_dir is None:
        raise FileNotFoundError(f"No external feature directory found for {structure_path.stem}.")
    return structure_dir_to_feature_lookup(feature_dir)


def load_structure_feature_sources(
    *,
    structure,
    structure_path: Path,
    structure_root: Path,
    embeddings_dir: Path,
    require_esm_embeddings: bool,
    feature_root_dir: Path,
    require_external_features: bool,
    feature_fallbacks: List[Dict[str, str]],
) -> StructureFeatureSources:
    esm_lookup: Dict[ResidueKey, Tensor] = {}
    if require_esm_embeddings or embeddings_dir.exists():
        try:
            esm_lookup = load_esm_lookup_for_structure(structure, structure_path, embeddings_dir)
        except FileNotFoundError as exc:
            if require_esm_embeddings:
                raise ValueError(f"Missing required ESM embeddings for {structure_path}: {exc}") from exc
            feature_fallbacks.append(
                feature_fallback_record(
                    structure_path,
                    feature_name="esm_embeddings",
                    detail=str(exc),
                )
            )
        except Exception as exc:
            raise ValueError(f"Invalid ESM embeddings for {structure_path}: {exc}") from exc

    try:
        external_feature_lookup = load_external_feature_lookup_for_structure(
            structure_path=structure_path,
            structure_root=structure_root,
            feature_root_dir=feature_root_dir,
        )
    except FileNotFoundError as exc:
        if require_external_features:
            raise ValueError(f"Missing required external features for {structure_path}: {exc}") from exc
        feature_fallbacks.append(
            feature_fallback_record(
                structure_path,
                feature_name="external_features",
                detail=str(exc),
            )
        )
        external_feature_lookup = {}
    except Exception as exc:
        raise ValueError(f"Invalid external features for {structure_path}: {exc}") from exc

    return StructureFeatureSources(
        esm_lookup=esm_lookup,
        external_feature_lookup=external_feature_lookup,
    )


def attach_structure_features_to_pocket(
    pocket: PocketRecord,
    *,
    feature_sources: StructureFeatureSources,
    esm_dim: int,
    require_esm_embeddings: bool,
    require_external_features: bool,
    structure_path: Path,
) -> None:
    try:
        if feature_sources.esm_lookup:
            attach_esm_embeddings(
                pocket,
                esm_lookup=feature_sources.esm_lookup,
                esm_dim=esm_dim,
                zero_if_missing=not require_esm_embeddings,
            )
        if feature_sources.external_feature_lookup:
            attach_external_residue_features(
                pocket,
                feature_sources.external_feature_lookup,
                strict=require_external_features,
            )
    except KeyError as exc:
        raise ValueError(f"Feature alignment error for {structure_path}: {exc}") from exc


def build_pocket_feature_coverage(pockets: List[PocketRecord]) -> Dict[str, float | int]:
    total_residues = sum(len(pocket.residues) for pocket in pockets)
    residues_with_esm = sum(
        1 for pocket in pockets for residue in pocket.residues if residue.has_esm_embedding
    )
    residues_with_external_features = sum(
        1 for pocket in pockets for residue in pocket.residues if residue.has_external_features
    )
    pockets_with_any_esm = sum(
        1 for pocket in pockets if any(residue.has_esm_embedding for residue in pocket.residues)
    )
    pockets_with_any_external = sum(
        1 for pocket in pockets if any(residue.has_external_features for residue in pocket.residues)
    )

    residue_denominator = max(1, total_residues)
    pocket_denominator = max(1, len(pockets))
    return {
        "total_pockets": len(pockets),
        "total_residues": total_residues,
        "residues_with_esm_embeddings": residues_with_esm,
        "residues_with_external_features": residues_with_external_features,
        "esm_residue_coverage": residues_with_esm / residue_denominator,
        "external_feature_residue_coverage": residues_with_external_features / residue_denominator,
        "pockets_with_any_esm_embeddings": pockets_with_any_esm,
        "pockets_with_any_external_features": pockets_with_any_external,
        "esm_pocket_coverage": pockets_with_any_esm / pocket_denominator,
        "external_feature_pocket_coverage": pockets_with_any_external / pocket_denominator,
    }


def build_feature_load_report(
    *,
    pockets: List[PocketRecord],
    total_structure_files: int,
    skipped_structures: List[Dict[str, str]],
    feature_fallbacks: List[Dict[str, str]],
    skipped_pockets: List[Dict[str, str]],
) -> Dict[str, object]:
    loaded_structures = {
        str(pocket.metadata.get("source_path", pocket.structure_id))
        for pocket in pockets
    }
    return {
        "total_structure_files": total_structure_files,
        "loaded_structure_files": len(loaded_structures),
        "skipped_structures": skipped_structures,
        "feature_fallbacks": feature_fallbacks,
        "skipped_pockets": skipped_pockets,
        **build_pocket_feature_coverage(pockets),
    }


__all__ = [
    "StructureFeatureSources",
    "attach_structure_features_to_pocket",
    "build_feature_load_report",
    "build_pocket_feature_coverage",
    "load_structure_feature_sources",
    "resolve_runtime_feature_paths",
]
