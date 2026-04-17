from __future__ import annotations

from pathlib import Path
from typing import Any

from data_structures import PocketRecord
from graph.construction import (
    canonical_ring_edges_output_path,
    extract_metal_pockets_from_structure,
    parse_structure_file,
)
from training.feature_sources import (
    attach_structure_features_to_pocket,
    build_feature_load_report,
    load_structure_feature_sources,
)
from training.labels import infer_metal_target_class_from_pocket, parse_ec_top_level_from_structure_path
from training.site_filter import SiteKey, pocket_matches_allowed_sites
def pocket_has_full_supervision(pocket: PocketRecord) -> bool:
    return pocket.y_metal is not None and pocket.y_ec is not None


def is_auxiliary_structure_file(path: Path, structure_root: Path) -> bool:
    try:
        relative_parts = path.relative_to(structure_root).parts
    except ValueError:
        relative_parts = path.parts

    if len(relative_parts) > 2:
        return True
    return path.parent.name == path.stem


def find_structure_files(structure_dir: Path) -> list[Path]:
    structure_files: list[Path] = []
    for pattern in ("*.pdb", "*.cif", "*.mmcif"):
        structure_files.extend(structure_dir.rglob(pattern))
    return sorted(
        path for path in structure_files if path.is_file() and not is_auxiliary_structure_file(path, structure_dir)
    )


def build_load_report(
    *,
    pockets: list[PocketRecord],
    structure_files: list[Path],
    feature_fallbacks: list[dict[str, str]],
    skipped_pockets: list[dict[str, str]],
) -> dict[str, Any]:
    return build_feature_load_report(
        pockets=pockets,
        total_structure_files=len(structure_files),
        feature_fallbacks=feature_fallbacks,
        skipped_pockets=skipped_pockets,
    )


def load_structure_pockets(
    *,
    structure_path: Path,
    structure_root: Path,
    allowed_site_keys: set[SiteKey] | None,
    esm_dim: int,
    embeddings_dir: Path,
    require_esm_embeddings: bool,
    feature_root_dir: Path,
    require_external_features: bool,
    unsupported_metal_policy: str = "error",
) -> tuple[list[PocketRecord], list[dict[str, str]], list[dict[str, str]]]:
    structure = parse_structure_file(str(structure_path), structure_id=structure_path.stem)
    extracted_pockets = extract_metal_pockets_from_structure(structure, structure_id=structure_path.stem)
    if not extracted_pockets:
        return [], [], []

    feature_fallbacks: list[dict[str, str]] = []
    skipped_pockets: list[dict[str, str]] = []
    feature_sources = load_structure_feature_sources(
        structure=structure,
        structure_path=structure_path,
        structure_root=structure_root,
        embeddings_dir=embeddings_dir,
        require_esm_embeddings=require_esm_embeddings,
        feature_root_dir=feature_root_dir,
        require_external_features=require_external_features,
        feature_fallbacks=feature_fallbacks,
    )
    ec_label = parse_ec_top_level_from_structure_path(structure_path)

    kept_pockets: list[PocketRecord] = []
    for pocket in extracted_pockets:
        pocket.metadata["source_path"] = str(structure_path)
        pocket.metadata.setdefault(
            "ring_edges_expected_path",
            str(canonical_ring_edges_output_path(structure_path)),
        )
        attach_structure_features_to_pocket(
            pocket,
            feature_sources=feature_sources,
            esm_dim=esm_dim,
            require_esm_embeddings=require_esm_embeddings,
            require_external_features=require_external_features,
            structure_path=structure_path,
        )

        if allowed_site_keys is not None and not pocket_matches_allowed_sites(
            pocket,
            structure_path,
            allowed_site_keys,
        ):
            skipped_pockets.append(
                {
                    "structure_id": pocket.structure_id,
                    "pocket_id": pocket.pocket_id,
                    "reason": "filtered_by_summary_sites",
                }
            )
            continue

        try:
            pocket.y_metal = infer_metal_target_class_from_pocket(
                pocket,
                unsupported_policy=unsupported_metal_policy,
            )
        except ValueError:
            skipped_pockets.append(
                {
                    "structure_id": pocket.structure_id,
                    "pocket_id": pocket.pocket_id,
                    "reason": "unsupported_metal_label",
                }
            )
            raise
        if pocket.y_metal is None and unsupported_metal_policy == "skip":
            skipped_pockets.append(
                {
                    "structure_id": pocket.structure_id,
                    "pocket_id": pocket.pocket_id,
                    "reason": "unsupported_metal_label",
                }
            )
            continue
        if ec_label is not None:
            pocket.y_ec = ec_label
        kept_pockets.append(pocket)

    return kept_pockets, feature_fallbacks, skipped_pockets
