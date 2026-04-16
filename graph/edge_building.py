from __future__ import annotations

"""Compatibility exports for graph edge utilities.

The implementation now lives in smaller focused modules:
- `graph.shell_roles`
- `graph.edge_geometry`
- `graph.edge_sources`
- `graph.edge_postprocess`
"""

from graph.edge_geometry import (
    build_pair_edge_geometry,
    build_radius_graph_from_residues,
    candidate_residue_pairs_within_radius,
    canonicalize_edge_pair,
    closest_points_between_coord_tensors,
    closest_points_between_residues,
    residue_atom_coords,
    residue_atom_coords_list,
    residue_spatial_envelope,
)
from graph.edge_postprocess import (
    expand_edge_records_bidirectionally,
    merge_edge_records,
    stack_edge_features,
)
from graph.edge_sources import (
    RING_METAL_CONTACT_POLICY,
    build_radius_edge_records_from_residues,
    build_ring_interaction_edge_records,
    radius_edge_records_from_index,
)
from graph.shell_roles import annotate_shell_roles, compute_shell_roles

__all__ = [
    "RING_METAL_CONTACT_POLICY",
    "annotate_shell_roles",
    "build_pair_edge_geometry",
    "build_radius_edge_records_from_residues",
    "build_radius_graph_from_residues",
    "build_ring_interaction_edge_records",
    "candidate_residue_pairs_within_radius",
    "canonicalize_edge_pair",
    "closest_points_between_coord_tensors",
    "closest_points_between_residues",
    "compute_shell_roles",
    "expand_edge_records_bidirectionally",
    "merge_edge_records",
    "radius_edge_records_from_index",
    "residue_atom_coords",
    "residue_atom_coords_list",
    "residue_spatial_envelope",
    "stack_edge_features",
]
