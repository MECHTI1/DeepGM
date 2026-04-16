from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from data_structures import EDGE_SOURCE_TO_INDEX


def _edge_merge_priority(record: dict[str, Any]) -> tuple[int, float]:
    source_type = record["source_type"]
    is_radius = float(source_type[EDGE_SOURCE_TO_INDEX["radius"]].item()) > 0.5
    contact_distance = float(record["dist_raw"][0].item())
    return (0 if is_radius else 1, contact_distance)


def merge_edge_records(edge_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged_by_pair: dict[tuple[int, int], dict[str, Any]] = {}
    for record in edge_records:
        edge_key = (int(record["src"]), int(record["dst"]))
        existing = merged_by_pair.get(edge_key)
        if existing is None:
            merged_by_pair[edge_key] = {
                "src": edge_key[0],
                "dst": edge_key[1],
                "dist_raw": record["dist_raw"].clone(),
                "seqsep": float(record["seqsep"]),
                "same_chain": float(record["same_chain"]),
                "vector_raw": record["vector_raw"].clone(),
                "interaction_type": record["interaction_type"].clone(),
                "source_type": record["source_type"].clone(),
            }
            continue

        should_replace_geometry = _edge_merge_priority(record) < _edge_merge_priority(existing)
        existing["interaction_type"] = torch.maximum(existing["interaction_type"], record["interaction_type"])
        existing["source_type"] = torch.maximum(existing["source_type"], record["source_type"])
        if should_replace_geometry:
            existing["dist_raw"] = record["dist_raw"].clone()
            existing["seqsep"] = float(record["seqsep"])
            existing["same_chain"] = float(record["same_chain"])
            existing["vector_raw"] = record["vector_raw"].clone()

    return [merged_by_pair[key] for key in sorted(merged_by_pair)]


def expand_edge_records_bidirectionally(edge_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    expanded_records: list[dict[str, Any]] = []
    for record in edge_records:
        expanded_records.append(
            {
                "src": int(record["src"]),
                "dst": int(record["dst"]),
                "dist_raw": record["dist_raw"].clone(),
                "seqsep": float(record["seqsep"]),
                "same_chain": float(record["same_chain"]),
                "vector_raw": record["vector_raw"].clone(),
                "interaction_type": record["interaction_type"].clone(),
                "source_type": record["source_type"].clone(),
            }
        )
        if int(record["src"]) == int(record["dst"]):
            continue
        expanded_records.append(
            {
                "src": int(record["dst"]),
                "dst": int(record["src"]),
                "dist_raw": record["dist_raw"].clone(),
                "seqsep": float(record["seqsep"]),
                "same_chain": float(record["same_chain"]),
                "vector_raw": (-record["vector_raw"]).clone(),
                "interaction_type": record["interaction_type"].clone(),
                "source_type": record["source_type"].clone(),
            }
        )
    return expanded_records


def stack_edge_features(edge_records: list[dict[str, Any]], bidirectional: bool = True) -> dict[str, Tensor]:
    stacked_records = expand_edge_records_bidirectionally(edge_records) if bidirectional else edge_records
    return {
        "edge_index": torch.tensor(
            [[record["src"] for record in stacked_records], [record["dst"] for record in stacked_records]],
            dtype=torch.long,
        ),
        "edge_dist_raw": torch.stack([record["dist_raw"] for record in stacked_records], dim=0),
        "edge_seqsep": torch.tensor([record["seqsep"] for record in stacked_records], dtype=torch.float32).unsqueeze(-1),
        "edge_same_chain": torch.tensor(
            [record["same_chain"] for record in stacked_records],
            dtype=torch.float32,
        ).unsqueeze(-1),
        "edge_vector_raw": torch.stack([record["vector_raw"] for record in stacked_records], dim=0),
        "edge_interaction_type": torch.stack([record["interaction_type"] for record in stacked_records], dim=0),
        "edge_source_type": torch.stack([record["source_type"] for record in stacked_records], dim=0),
    }
