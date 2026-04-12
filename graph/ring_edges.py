from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import torch
from torch import Tensor

from data_structures import PocketRecord, ResidueRecord
from project_paths import get_default_embeddings_dir


DEFAULT_RING_OUTPUT_DIR = get_default_embeddings_dir()


def parse_ring_node_id(node_id: str) -> Tuple[str, int, str, str]:
    parts = node_id.strip().split(":")
    if len(parts) != 4:
        raise ValueError(f"Unsupported ring node id format: {node_id!r}")

    chain_id, resseq_text, icode, resname = parts
    return chain_id, int(resseq_text), "" if icode in {"_", ".", "?"} else icode, resname


def parse_embedded_coord(text: str) -> Optional[Tensor]:
    raw = text.strip()
    if not raw:
        return None

    parts = [part.strip() for part in raw.split(",")]
    if len(parts) != 3:
        return None

    try:
        values = [float(part) for part in parts]
    except ValueError:
        return None
    return torch.tensor(values, dtype=torch.float32)


def resolve_ring_endpoint_coord(residue: ResidueRecord, atom_or_coord: str) -> Optional[Tensor]:
    coord = parse_embedded_coord(atom_or_coord)
    if coord is not None:
        return coord

    atom_name = atom_or_coord.strip()
    if not atom_name:
        return None

    atom = residue.get_atom(atom_name)
    return atom.float() if atom is not None else None


def canonical_ring_edges_output_path(structure_path: str | Path) -> Path:
    structure_path = Path(structure_path)
    return DEFAULT_RING_OUTPUT_DIR / structure_path.stem / f"{structure_path.name}_ringEdges"


def ring_edges_path_candidates(
    structure_id: str,
    source_path: Optional[str] = None,
    explicit_path: Optional[str] = None,
    expected_path: Optional[str] = None,
) -> List[Path]:
    candidates: List[Path] = []
    seen = set()

    def add_candidate(path: Path) -> None:
        key = str(path)
        if key in seen:
            return
        seen.add(key)
        candidates.append(path)

    for maybe_path in (explicit_path, expected_path):
        if maybe_path:
            add_candidate(Path(maybe_path))

    if source_path:
        source = Path(source_path)
        add_candidate(canonical_ring_edges_output_path(source))
        add_candidate(Path(f"{source_path}_ringEdges"))

    normalized_structure_id = structure_id.strip()
    if normalized_structure_id:
        embedding_dir = DEFAULT_RING_OUTPUT_DIR / normalized_structure_id
        add_candidate(embedding_dir / f"{normalized_structure_id}.pdb_ringEdges")
        add_candidate(embedding_dir / f"{normalized_structure_id}.cif_ringEdges")
        if embedding_dir.is_dir():
            for candidate in sorted(embedding_dir.glob("*ringEdges")):
                add_candidate(candidate)

    return candidates


def resolve_ring_edges_path(pocket: PocketRecord) -> Optional[Path]:
    for candidate in ring_edges_path_candidates(
        structure_id=pocket.structure_id,
        source_path=pocket.metadata.get("source_path"),
        explicit_path=pocket.metadata.get("ring_edges_path"),
        expected_path=pocket.metadata.get("ring_edges_expected_path"),
    ):
        if candidate.is_file():
            pocket.metadata["ring_edges_path"] = str(candidate)
            return candidate
    return None
