from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from graph.ring_edges import canonical_ring_edges_output_path, ring_edges_path_candidates
from project_paths import resolve_embeddings_dir
from training.esm_feature_loading import embedding_path_candidates
from training.structure_loading import find_structure_files


def _structure_has_esm_embedding(structure_path: Path, embeddings_dir: Path) -> bool:
    return any(candidate.is_file() for candidate in embedding_path_candidates(embeddings_dir, structure_path))


def _structure_has_ring_edges(structure_path: Path) -> bool:
    return any(
        candidate.is_file()
        for candidate in ring_edges_path_candidates(
            structure_id=structure_path.stem,
            source_path=str(structure_path),
            expected_path=str(canonical_ring_edges_output_path(structure_path)),
        )
    )


def discover_missing_esm_embeddings(
    structure_files: Sequence[Path],
    embeddings_dir: Path,
) -> list[Path]:
    return [structure_path for structure_path in structure_files if not _structure_has_esm_embedding(structure_path, embeddings_dir)]


def discover_missing_ring_edges(structure_files: Sequence[Path]) -> list[Path]:
    return [structure_path for structure_path in structure_files if not _structure_has_ring_edges(structure_path)]


def _generate_missing_esm_embeddings(
    structure_files: Sequence[Path],
    embeddings_dir: Path,
) -> dict[str, object]:
    try:
        from embed_helpers.esmc import create_resi_embed_batch
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependencies for real ESM embedding generation. "
            "Install the packages required by embed_helpers/esmc.py."
        ) from exc

    return create_resi_embed_batch(structure_files, out_dir=embeddings_dir, overwrite=False)


def _generate_missing_ring_edges(
    structure_files: Sequence[Path],
    embeddings_dir: Path,
) -> dict[str, object]:
    from embed_helpers.Interaction_edge import create_ring_edges_batch

    return create_ring_edges_batch(structure_files, dir_results=embeddings_dir, overwrite=False)


def _raise_on_failed_generation(
    *,
    summary: dict[str, object],
    feature_name: str,
) -> None:
    failed_structures = list(summary.get("failed_structures", []))
    if not failed_structures:
        return
    raise ValueError(
        f"Failed to generate missing {feature_name} for {len(failed_structures)} structure(s). "
        f"Sample: {failed_structures[:3]}"
    )


def prepare_runtime_inputs(
    *,
    structure_dir: Path,
    esm_embeddings_dir: str | Path | None,
    require_esm_embeddings: bool,
    prepare_missing_esm_embeddings: bool,
    require_ring_edges: bool,
    prepare_missing_ring_edges: bool,
) -> dict[str, Any]:
    structure_files = find_structure_files(Path(structure_dir))
    embeddings_dir = resolve_embeddings_dir(
        str(esm_embeddings_dir) if esm_embeddings_dir is not None else None,
        create=True,
    )

    report: dict[str, Any] = {
        "total_structure_files": len(structure_files),
        "esm_embeddings_dir": str(embeddings_dir),
        "missing_esm_structures_before": 0,
        "generated_esm_files": 0,
        "missing_ring_edge_structures_before": 0,
        "generated_ring_edge_files": 0,
    }

    if not structure_files:
        return report

    should_prepare_esm = require_esm_embeddings and prepare_missing_esm_embeddings
    if should_prepare_esm:
        missing_esm_structures = discover_missing_esm_embeddings(structure_files, embeddings_dir)
        report["missing_esm_structures_before"] = len(missing_esm_structures)
        if missing_esm_structures:
            summary = _generate_missing_esm_embeddings(missing_esm_structures, embeddings_dir)
            _raise_on_failed_generation(summary=summary, feature_name="ESM embeddings")
            report["generated_esm_files"] = len(list(summary.get("saved_files", [])))

    should_prepare_ring_edges = require_ring_edges or prepare_missing_ring_edges
    if should_prepare_ring_edges:
        missing_ring_structures = discover_missing_ring_edges(structure_files)
        report["missing_ring_edge_structures_before"] = len(missing_ring_structures)
        if missing_ring_structures:
            summary = _generate_missing_ring_edges(missing_ring_structures, embeddings_dir)
            _raise_on_failed_generation(summary=summary, feature_name="RING edge files")
            report["generated_ring_edge_files"] = len(list(summary.get("saved_files", [])))

    return report
