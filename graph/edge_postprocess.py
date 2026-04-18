from __future__ import annotations

import torch
from torch import Tensor

from data_structures import EDGE_SOURCE_TO_INDEX, EDGE_SOURCE_TYPES, INTERACTION_SUMMARIES_OPTIONAL_WITH_RING
from graph.edge_records import ResidueEdgeRecord, ResidueMetalEdgeRecord


def _edge_merge_priority(record: ResidueEdgeRecord) -> tuple[int, float]:
    source_type = record.source_type
    is_radius = float(source_type[EDGE_SOURCE_TO_INDEX["radius"]].item()) > 0.5
    contact_distance = float(record.dist_raw[0].item())
    return (0 if is_radius else 1, contact_distance)


def merge_edge_records(edge_records: list[ResidueEdgeRecord]) -> list[ResidueEdgeRecord]:
    merged_by_pair: dict[tuple[int, int], ResidueEdgeRecord] = {}
    for record in edge_records:
        edge_key = (int(record.src), int(record.dst))
        existing = merged_by_pair.get(edge_key)
        if existing is None:
            merged_by_pair[edge_key] = record.clone()
            continue

        should_replace_geometry = _edge_merge_priority(record) < _edge_merge_priority(existing)
        existing.interaction_type = torch.maximum(existing.interaction_type, record.interaction_type)
        existing.source_type = torch.maximum(existing.source_type, record.source_type)
        if should_replace_geometry:
            existing.dist_raw = record.dist_raw.clone()
            existing.seqsep = float(record.seqsep)
            existing.same_chain = float(record.same_chain)
            existing.vector_raw = record.vector_raw.clone()
            existing.geometry_label = record.geometry_label

    return [merged_by_pair[key] for key in sorted(merged_by_pair)]


def expand_edge_records_bidirectionally(edge_records: list[ResidueEdgeRecord]) -> list[ResidueEdgeRecord]:
    expanded_records: list[ResidueEdgeRecord] = []
    for record in edge_records:
        expanded_records.append(record.clone())
        if int(record.src) == int(record.dst):
            continue
        expanded_records.append(record.reversed_copy())
    return expanded_records


def stack_edge_features(edge_records: list[ResidueEdgeRecord], bidirectional: bool = True) -> dict[str, Tensor]:
    stacked_records = expand_edge_records_bidirectionally(edge_records) if bidirectional else edge_records
    if not stacked_records:
        return {
            "edge_index": torch.zeros((2, 0), dtype=torch.long),
            "edge_dist_raw": torch.zeros((0, 2), dtype=torch.float32),
            "edge_seqsep": torch.zeros((0, 1), dtype=torch.float32),
            "edge_same_chain": torch.zeros((0, 1), dtype=torch.float32),
            "edge_vector_raw": torch.zeros((0, 3), dtype=torch.float32),
            "edge_interaction_type": torch.zeros((0, len(INTERACTION_SUMMARIES_OPTIONAL_WITH_RING)), dtype=torch.float32),
            "edge_source_type": torch.zeros((0, len(EDGE_SOURCE_TYPES)), dtype=torch.float32),
        }
    return {
        "edge_index": torch.tensor(
            [[record.src for record in stacked_records], [record.dst for record in stacked_records]],
            dtype=torch.long,
        ),
        "edge_dist_raw": torch.stack([record.dist_raw for record in stacked_records], dim=0),
        "edge_seqsep": torch.tensor([record.seqsep for record in stacked_records], dtype=torch.float32).unsqueeze(-1),
        "edge_same_chain": torch.tensor(
            [record.same_chain for record in stacked_records],
            dtype=torch.float32,
        ).unsqueeze(-1),
        "edge_vector_raw": torch.stack([record.vector_raw for record in stacked_records], dim=0),
        "edge_interaction_type": torch.stack([record.interaction_type for record in stacked_records], dim=0),
        "edge_source_type": torch.stack([record.source_type for record in stacked_records], dim=0),
    }


def stack_metal_edge_features(edge_records: list[ResidueMetalEdgeRecord]) -> dict[str, Tensor]:
    if not edge_records:
        return {
            "metal_edge_index": torch.zeros((2, 0), dtype=torch.long),
            "metal_edge_dist_raw": torch.zeros((0, 1), dtype=torch.float32),
            "metal_edge_vector_raw": torch.zeros((0, 3), dtype=torch.float32),
            "metal_edge_interaction_type": torch.zeros((0, len(INTERACTION_SUMMARIES_OPTIONAL_WITH_RING)), dtype=torch.float32),
            "metal_edge_source_type": torch.zeros((0, len(EDGE_SOURCE_TYPES)), dtype=torch.float32),
        }

    return {
        "metal_edge_index": torch.tensor(
            [[record.residue_idx for record in edge_records], [record.metal_idx for record in edge_records]],
            dtype=torch.long,
        ),
        "metal_edge_dist_raw": torch.stack([record.dist_raw for record in edge_records], dim=0),
        "metal_edge_vector_raw": torch.stack([record.vector_raw for record in edge_records], dim=0),
        "metal_edge_interaction_type": torch.stack([record.interaction_type for record in edge_records], dim=0),
        "metal_edge_source_type": torch.stack([record.source_type for record in edge_records], dim=0),
    }
