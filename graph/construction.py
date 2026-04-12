from __future__ import annotations

import json
from typing import Dict, List

import torch
from torch import Tensor
from torch_geometric.data import Data

from data_structures import (
    DEFAULT_EDGE_RADIUS,
    PocketRecord,
)
from featurization import (
    MultinuclearSiteHandler,
    compute_net_ligand_vector,
    residue_to_stage1_node_features,
)
from graph.edge_building import (
    annotate_shell_roles,
    build_pair_edge_geometry,
    build_radius_graph_from_residues,
    build_ring_interaction_edge_records,
    radius_edge_records_from_index,
    stack_edge_features,
)
from graph.feature_utils import (
    attach_esm_embeddings,
    attach_external_residue_features,
)
from graph.ring_edges import (
    canonical_ring_edges_output_path,
)
from graph.structure_parsing import (
    extract_metal_pockets_from_structure,
    parse_structure_file,
)

__all__ = [
    "attach_esm_embeddings",
    "attach_external_residue_features",
    "annotate_shell_roles",
    "build_pair_edge_geometry",
    "build_radius_graph_from_residues",
    "build_ring_interaction_edge_records",
    "canonical_ring_edges_output_path",
    "extract_metal_pockets_from_structure",
    "parse_structure_file",
    "pocket_to_pyg_data",
    "save_pocket_metadata_json",
]


def stack_node_features(node_dicts: List[Dict[str, Tensor]]) -> Dict[str, Tensor]:
    return {
        "x_esm": torch.stack([node["x_esm"] for node in node_dicts], dim=0),
        "x_reschem": torch.stack([node["x_reschem"] for node in node_dicts], dim=0),
        "x_role": torch.stack([node["x_role"] for node in node_dicts], dim=0),
        "x_dist_raw": torch.stack([node["x_dist_raw"] for node in node_dicts], dim=0),
        "x_misc": torch.stack([node["x_misc"] for node in node_dicts], dim=0),
        "x_env_burial": torch.stack([node["x_env_burial"] for node in node_dicts], dim=0),
        "x_env_pka": torch.stack([node["x_env_pka"] for node in node_dicts], dim=0),
        "x_env_conf": torch.stack([node["x_env_conf"] for node in node_dicts], dim=0),
        "x_env_interactions": torch.stack([node["x_env_interactions"] for node in node_dicts], dim=0),
        "x_vec": torch.stack([node["x_vec"] for node in node_dicts], dim=0),
        "donor_coords": torch.stack([node["donor_coords"] for node in node_dicts], dim=0),
        "donor_mask": torch.stack([node["donor_mask"] for node in node_dicts], dim=0),
        "fg_centroid": torch.stack([node["fg_centroid"] for node in node_dicts], dim=0),
        "pos": torch.stack([node["pos"] for node in node_dicts], dim=0),
    }


def pocket_to_pyg_data(
    pocket: PocketRecord,
    esm_dim: int,
    edge_radius: float = DEFAULT_EDGE_RADIUS,
    require_ring_edges: bool = False,
) -> Data:
    annotate_shell_roles(pocket)
    v_net = compute_net_ligand_vector(pocket)
    node_features = stack_node_features(
        [residue_to_stage1_node_features(residue, pocket, esm_dim, v_net) for residue in pocket.residues]
    )

    base_edge_index = build_radius_graph_from_residues(pocket.residues, edge_radius)
    edge_records = radius_edge_records_from_index(pocket, base_edge_index)
    edge_records.extend(build_ring_interaction_edge_records(pocket, require_ring_edges=require_ring_edges))
    if not edge_records:
        raise ValueError(
            f"Pocket {pocket.pocket_id} produced a graph with no edges at edge_radius={edge_radius}. "
            "Increase the radius, inspect the pocket residues, or provide ring interaction edges."
        )
    edge_features = stack_edge_features(edge_records)

    data = Data(
        **node_features,
        **edge_features,
        metal_pos=MultinuclearSiteHandler.metal_coords_for_pocket(pocket),
        metal_center_pos=pocket.metal_coord.unsqueeze(0),
        metal_count=torch.tensor([pocket.metal_count()], dtype=torch.long),
        is_multinuclear=torch.tensor([int(pocket.is_multinuclear())], dtype=torch.long),
        site_metal_stats=MultinuclearSiteHandler.site_metal_stats(pocket).unsqueeze(0),
    )
    if pocket.y_metal is not None:
        data.y_metal = torch.tensor([pocket.y_metal], dtype=torch.long)
    if pocket.y_ec is not None:
        data.y_ec = torch.tensor([pocket.y_ec], dtype=torch.long)
    return data


def save_pocket_metadata_json(pocket: PocketRecord, outpath: str) -> None:
    payload = {
        "structure_id": pocket.structure_id,
        "pocket_id": pocket.pocket_id,
        "metal_element": pocket.metal_element,
        "metal_coord": pocket.metal_coord.tolist(),
        "metal_coords": [coord.tolist() for coord in pocket.metal_coords],
        "metal_count": pocket.metal_count(),
        "is_multinuclear": pocket.is_multinuclear(),
        "y_metal": pocket.y_metal,
        "y_ec": pocket.y_ec,
        "residues": [
            {
                "chain_id": residue.chain_id,
                "resseq": residue.resseq,
                "icode": residue.icode,
                "resname": residue.resname,
                "is_first_shell": residue.is_first_shell,
                "is_second_shell": residue.is_second_shell,
                "has_esm_embedding": residue.has_esm_embedding,
                "has_external_features": residue.has_external_features,
                "external_features": residue.external_features,
                "atom_names": sorted(list(residue.atoms.keys())),
            }
            for residue in pocket.residues
        ],
    }
    with open(outpath, "w") as handle:
        json.dump(payload, handle, indent=2)
