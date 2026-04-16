from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch

from data_structures import EDGE_SOURCE_TO_INDEX, PocketRecord, RING_INTERACTION_TO_INDEX, ResidueRecord
from graph.construction import pocket_to_pyg_data, save_pocket_metadata_json
from graph.edge_building import (
    build_pair_edge_geometry,
    build_radius_edge_records_from_residues,
    build_radius_graph_from_residues,
    build_ring_interaction_edge_records,
    candidate_residue_pairs_within_radius,
    compute_shell_roles,
    expand_edge_records_bidirectionally,
    merge_edge_records,
    radius_edge_records_from_index,
    stack_edge_features,
)
from graph.feature_utils import attach_esm_embeddings, attach_external_residue_features
from graph.structure_parsing import MetalAtomRecord, cluster_metal_records
from training.graph_dataset import apply_feature_normalization, compute_feature_normalization_stats


def make_residue(
    *,
    chain_id: str,
    resseq: int,
    ca: tuple[float, float, float],
    cb: tuple[float, float, float] | None = None,
    extra_atoms: dict[str, tuple[float, float, float]] | None = None,
) -> ResidueRecord:
    atoms = {"CA": torch.tensor(ca, dtype=torch.float32)}
    if cb is not None:
        atoms["CB"] = torch.tensor(cb, dtype=torch.float32)
    if extra_atoms is not None:
        for atom_name, coord in extra_atoms.items():
            atoms[atom_name] = torch.tensor(coord, dtype=torch.float32)
    return ResidueRecord(
        chain_id=chain_id,
        resseq=resseq,
        icode="",
        resname="HIS",
        atoms=atoms,
    )


def make_pocket(residues: list[ResidueRecord]) -> PocketRecord:
    return PocketRecord(
        structure_id="1abc__chain_A__EC_1.1.1.1",
        pocket_id="pocket-1",
        metal_element="ZN",
        metal_coords=[torch.tensor([0.0, 0.0, 0.0])],
        residues=residues,
    )


class GraphFeatureUtilsTests(unittest.TestCase):
    def test_attach_esm_embeddings_sets_zero_vector_when_missing(self) -> None:
        residue = make_residue(chain_id="A", resseq=1, ca=(0.0, 0.0, 0.0))
        pocket = make_pocket([residue])

        attach_esm_embeddings(pocket, esm_lookup={}, esm_dim=4, zero_if_missing=True)

        self.assertEqual(tuple(pocket.residues[0].esm_embedding.shape), (4,))
        self.assertFalse(pocket.residues[0].has_esm_embedding)
        self.assertEqual(float(pocket.residues[0].esm_embedding.sum().item()), 0.0)

    def test_attach_esm_embeddings_rejects_dimension_mismatch(self) -> None:
        residue = make_residue(chain_id="A", resseq=1, ca=(0.0, 0.0, 0.0))
        pocket = make_pocket([residue])

        with self.assertRaisesRegex(ValueError, "ESM embedding dimension mismatch"):
            attach_esm_embeddings(
                pocket,
                esm_lookup={("A", 1, ""): torch.ones(3, dtype=torch.float32)},
                esm_dim=4,
                zero_if_missing=True,
            )

    def test_attach_external_residue_features_marks_present_lookup(self) -> None:
        residue = make_residue(chain_id="A", resseq=2, ca=(0.0, 0.0, 0.0))
        pocket = make_pocket([residue])

        attach_external_residue_features(
            pocket,
            feature_lookup={("A", 2, ""): {"SASA": 12.5}},
            strict=False,
        )

        self.assertTrue(pocket.residues[0].has_external_features)
        self.assertEqual(pocket.residues[0].external_features["SASA"], 12.5)


class GraphEdgeBuildingTests(unittest.TestCase):
    def test_build_pair_edge_geometry_uses_contact_and_ca_distances(self) -> None:
        src = make_residue(chain_id="A", resseq=5, ca=(0.0, 0.0, 0.0), cb=(1.0, 0.0, 0.0))
        dst = make_residue(chain_id="A", resseq=8, ca=(0.0, 3.0, 0.0), cb=(1.0, 1.0, 0.0))

        edge_dist_raw, edge_seqsep, edge_same_chain, vector_raw = build_pair_edge_geometry(src, dst)

        self.assertEqual(tuple(edge_dist_raw.shape), (2,))
        self.assertAlmostEqual(edge_dist_raw[0].item(), 1.0, places=5)
        self.assertAlmostEqual(edge_dist_raw[1].item(), 3.0, places=5)
        self.assertEqual(edge_seqsep, 3.0)
        self.assertEqual(edge_same_chain, 1.0)
        self.assertTrue(torch.allclose(vector_raw, torch.tensor([0.0, 1.0, 0.0])))

    def test_build_radius_graph_from_residues_returns_canonical_pairs_within_cutoff(self) -> None:
        residues = [
            make_residue(chain_id="A", resseq=1, ca=(0.0, 0.0, 0.0)),
            make_residue(chain_id="A", resseq=2, ca=(0.0, 0.0, 2.0)),
            make_residue(chain_id="A", resseq=3, ca=(0.0, 0.0, 6.0)),
        ]

        edge_index = build_radius_graph_from_residues(residues, radius=2.5)

        self.assertEqual(edge_index.tolist(), [[0], [1]])

    def test_candidate_pairs_keep_sidechain_contacts_with_distant_ca_atoms(self) -> None:
        residues = [
            make_residue(
                chain_id="A",
                resseq=1,
                ca=(0.0, 0.0, 0.0),
                extra_atoms={"CB": (9.8, 0.0, 0.0)},
            ),
            make_residue(
                chain_id="A",
                resseq=2,
                ca=(20.0, 0.0, 0.0),
                extra_atoms={"CB": (10.2, 0.0, 0.0)},
            ),
        ]

        candidate_pairs = candidate_residue_pairs_within_radius(residues, radius=1.0)
        edge_index = build_radius_graph_from_residues(residues, radius=1.0)

        self.assertEqual(candidate_pairs, [(0, 1)])
        self.assertEqual(edge_index.tolist(), [[0], [1]])

    def test_build_radius_edge_records_matches_index_based_path(self) -> None:
        residues = [
            make_residue(chain_id="A", resseq=1, ca=(0.0, 0.0, 0.0), cb=(1.0, 0.0, 0.0)),
            make_residue(chain_id="A", resseq=2, ca=(0.0, 0.0, 2.0), cb=(1.0, 0.0, 2.0)),
        ]
        pocket = make_pocket(residues)

        edge_index = build_radius_graph_from_residues(residues, radius=2.5)
        expected = radius_edge_records_from_index(pocket, edge_index)
        actual = build_radius_edge_records_from_residues(pocket, radius=2.5)

        self.assertEqual(
            [(record["src"], record["dst"]) for record in actual],
            [(record["src"], record["dst"]) for record in expected],
        )
        self.assertTrue(
            all(torch.allclose(left["dist_raw"], right["dist_raw"]) for left, right in zip(actual, expected))
        )
        self.assertTrue(
            all(torch.allclose(left["vector_raw"], right["vector_raw"]) for left, right in zip(actual, expected))
        )

    def test_stack_edge_features_builds_tensor_batch(self) -> None:
        edge_features = stack_edge_features(
            [
                {
                    "src": 0,
                    "dst": 1,
                    "dist_raw": torch.tensor([1.0, 2.0]),
                    "seqsep": 4.0,
                    "same_chain": 1.0,
                    "vector_raw": torch.tensor([0.0, 1.0, 0.0]),
                    "interaction_type": torch.tensor([1.0, 0.0]),
                    "source_type": torch.tensor([0.0, 1.0]),
                }
            ],
            bidirectional=False,
        )

        self.assertEqual(edge_features["edge_index"].tolist(), [[0], [1]])
        self.assertEqual(tuple(edge_features["edge_dist_raw"].shape), (1, 2))
        self.assertEqual(tuple(edge_features["edge_seqsep"].shape), (1, 1))
        self.assertEqual(tuple(edge_features["edge_source_type"].shape), (1, 2))

    def test_expand_edge_records_bidirectionally_adds_reverse_messages(self) -> None:
        expanded = expand_edge_records_bidirectionally(
            [
                {
                    "src": 0,
                    "dst": 1,
                    "dist_raw": torch.tensor([1.0, 2.0]),
                    "seqsep": 4.0,
                    "same_chain": 1.0,
                    "vector_raw": torch.tensor([0.0, 1.0, 0.0]),
                    "interaction_type": torch.tensor([1.0, 0.0]),
                    "source_type": torch.tensor([0.0, 1.0]),
                }
            ]
        )

        self.assertEqual([(record["src"], record["dst"]) for record in expanded], [(0, 1), (1, 0)])
        self.assertTrue(torch.allclose(expanded[0]["vector_raw"], torch.tensor([0.0, 1.0, 0.0])))
        self.assertTrue(torch.allclose(expanded[1]["vector_raw"], torch.tensor([0.0, -1.0, 0.0])))

    def test_merge_edge_records_combines_sources_per_pair(self) -> None:
        merged = merge_edge_records(
            [
                {
                    "src": 0,
                    "dst": 1,
                    "dist_raw": torch.tensor([1.0, 2.0]),
                    "seqsep": 4.0,
                    "same_chain": 1.0,
                    "vector_raw": torch.tensor([0.0, 1.0, 0.0]),
                    "interaction_type": torch.tensor([0.0, 0.0, 0.0]),
                    "source_type": torch.tensor([1.0, 0.0]),
                },
                {
                    "src": 0,
                    "dst": 1,
                    "dist_raw": torch.tensor([1.2, 2.0]),
                    "seqsep": 4.0,
                    "same_chain": 1.0,
                    "vector_raw": torch.tensor([0.0, 0.8, 0.0]),
                    "interaction_type": torch.tensor([0.0, 1.0, 0.0]),
                    "source_type": torch.tensor([0.0, 1.0]),
                },
            ]
        )

        self.assertEqual([(record["src"], record["dst"]) for record in merged], [(0, 1)])
        self.assertTrue(torch.allclose(merged[0]["source_type"], torch.tensor([1.0, 1.0])))
        self.assertTrue(torch.allclose(merged[0]["interaction_type"], torch.tensor([0.0, 1.0, 0.0])))
        self.assertTrue(torch.allclose(merged[0]["dist_raw"], torch.tensor([1.0, 2.0])))

    def test_merge_edge_records_prefers_radius_geometry_independent_of_input_order(self) -> None:
        merged = merge_edge_records(
            [
                {
                    "src": 0,
                    "dst": 1,
                    "dist_raw": torch.tensor([0.5, 2.0]),
                    "seqsep": 1.0,
                    "same_chain": 1.0,
                    "vector_raw": torch.tensor([0.0, 0.5, 0.0]),
                    "interaction_type": torch.tensor([0.0, 1.0]),
                    "source_type": torch.tensor([0.0, 1.0]),
                },
                {
                    "src": 0,
                    "dst": 1,
                    "dist_raw": torch.tensor([1.5, 2.0]),
                    "seqsep": 1.0,
                    "same_chain": 1.0,
                    "vector_raw": torch.tensor([0.0, 1.5, 0.0]),
                    "interaction_type": torch.tensor([0.0, 0.0]),
                    "source_type": torch.tensor([1.0, 0.0]),
                },
            ]
        )

        self.assertTrue(torch.allclose(merged[0]["source_type"], torch.tensor([1.0, 1.0])))
        self.assertTrue(torch.allclose(merged[0]["dist_raw"], torch.tensor([1.5, 2.0])))
        self.assertTrue(torch.allclose(merged[0]["vector_raw"], torch.tensor([0.0, 1.5, 0.0])))

    def test_build_ring_interaction_edge_records_canonicalizes_residue_edges_and_keeps_metal_contacts(self) -> None:
        residues = [
            make_residue(
                chain_id="A",
                resseq=1,
                ca=(0.0, 0.0, 0.0),
                cb=(0.8, 0.5, 0.0),
                extra_atoms={"ND1": (0.4, 0.2, 0.0)},
            ),
            make_residue(
                chain_id="A",
                resseq=2,
                ca=(0.0, 0.0, 2.0),
                cb=(0.7, 0.4, 2.1),
                extra_atoms={"ND1": (0.2, 0.1, 1.5)},
            ),
        ]
        pocket = make_pocket(residues)
        pocket.metadata["metal_site_coord_map"] = {("A", 500, ""): torch.tensor([0.0, 0.0, 0.8], dtype=torch.float32)}

        with tempfile.TemporaryDirectory() as tmpdir:
            ring_path = Path(tmpdir) / "sample_ringEdges"
            ring_path.write_text(
                "\t".join(["NodeId1", "NodeId2", "Interaction", "Atom1", "Atom2"]) + "\n"
                + "\t".join(["A:1:_:HIS", "A:2:_:HIS", "HBOND:SC_SC", "ND1", "ND1"]) + "\n"
                + "\t".join(["A:1:_:HIS", "A:500:_:ZN", "METAL_ION:SC_LIG", "ND1", ""]) + "\n",
                encoding="utf-8",
            )
            pocket.metadata["ring_edges_path"] = str(ring_path)

            edge_records = build_ring_interaction_edge_records(pocket)

        edge_pairs = {(record["src"], record["dst"]) for record in edge_records}
        interaction_indices = {int(torch.argmax(record["interaction_type"]).item()) for record in edge_records}
        self.assertEqual(edge_pairs, {(0, 1), (0, 0)})
        self.assertIn(RING_INTERACTION_TO_INDEX["HBOND:SC_SC"], interaction_indices)
        self.assertIn(RING_INTERACTION_TO_INDEX["METAL_ION:SC_LIG"], interaction_indices)

    def test_pocket_to_pyg_data_merges_edge_sources_per_pair(self) -> None:
        residues = [
            make_residue(
                chain_id="A",
                resseq=1,
                ca=(0.0, 0.0, 0.0),
                cb=(0.8, 0.5, 0.0),
                extra_atoms={"ND1": (0.4, 0.2, 0.0)},
            ),
            make_residue(
                chain_id="A",
                resseq=2,
                ca=(0.0, 0.0, 2.0),
                cb=(0.7, 0.4, 2.1),
                extra_atoms={"ND1": (0.2, 0.1, 1.5)},
            ),
        ]
        for residue in residues:
            residue.esm_embedding = torch.ones(4, dtype=torch.float32)
            residue.has_esm_embedding = True
            residue.external_features = {"SASA": 1.0}
            residue.has_external_features = True
        pocket = make_pocket(residues)
        pocket.metadata["metal_site_coord_map"] = {("A", 500, ""): torch.tensor([0.0, 0.0, 0.8], dtype=torch.float32)}

        with tempfile.TemporaryDirectory() as tmpdir:
            ring_path = Path(tmpdir) / "sample_ringEdges"
            ring_path.write_text(
                "\t".join(["NodeId1", "NodeId2", "Interaction", "Atom1", "Atom2"]) + "\n"
                + "\t".join(["A:1:_:HIS", "A:2:_:HIS", "HBOND:SC_SC", "ND1", "ND1"]) + "\n"
                + "\t".join(["A:1:_:HIS", "A:500:_:ZN", "METAL_ION:SC_LIG", "ND1", ""]) + "\n",
                encoding="utf-8",
            )
            pocket.metadata["ring_edges_path"] = str(ring_path)

            data = pocket_to_pyg_data(pocket, esm_dim=4, edge_radius=3.0)

        ring_mask = data.edge_source_type[:, EDGE_SOURCE_TO_INDEX["ring"]] > 0.5
        radius_mask = data.edge_source_type[:, EDGE_SOURCE_TO_INDEX["radius"]] > 0.5
        self.assertGreater(int(ring_mask.sum().item()), 0)
        self.assertIn([0, 0], data.edge_index.t().tolist())
        self.assertIn([0, 1], data.edge_index.t().tolist())
        self.assertIn([1, 0], data.edge_index.t().tolist())
        edge_rows = data.edge_index.t().tolist()
        forward_idx = edge_rows.index([0, 1])
        reverse_idx = edge_rows.index([1, 0])
        self.assertTrue(bool(ring_mask[forward_idx].item()))
        self.assertTrue(bool(radius_mask[forward_idx].item()))
        self.assertTrue(bool(ring_mask[reverse_idx].item()))
        self.assertTrue(bool(radius_mask[reverse_idx].item()))

    def test_pocket_to_pyg_data_does_not_mutate_residue_shell_flags(self) -> None:
        residues = [
            make_residue(
                chain_id="A",
                resseq=1,
                ca=(0.0, 0.0, 0.0),
                cb=(0.8, 0.5, 0.0),
                extra_atoms={"ND1": (0.4, 0.2, 0.0)},
            ),
            make_residue(
                chain_id="A",
                resseq=2,
                ca=(0.0, 0.0, 2.0),
                cb=(0.7, 0.4, 2.1),
                extra_atoms={"ND1": (0.2, 0.1, 1.5)},
            ),
        ]
        for residue in residues:
            residue.esm_embedding = torch.ones(4, dtype=torch.float32)
            residue.has_esm_embedding = True
            residue.external_features = {"SASA": 1.0}
            residue.has_external_features = True
        pocket = make_pocket(residues)

        expected_shell_roles = compute_shell_roles(pocket)
        self.assertEqual([(r.is_first_shell, r.is_second_shell) for r in pocket.residues], [(False, False), (False, False)])

        data = pocket_to_pyg_data(pocket, esm_dim=4, edge_radius=3.0)

        self.assertEqual([(r.is_first_shell, r.is_second_shell) for r in pocket.residues], [(False, False), (False, False)])
        self.assertEqual(data.x_role.tolist(), [[float(a), float(b)] for a, b in expected_shell_roles])

    def test_save_pocket_metadata_json_uses_computed_shell_roles(self) -> None:
        residues = [
            make_residue(
                chain_id="A",
                resseq=1,
                ca=(0.0, 0.0, 0.0),
                cb=(0.8, 0.5, 0.0),
                extra_atoms={"ND1": (0.4, 0.2, 0.0)},
            )
        ]
        pocket = make_pocket(residues)

        with tempfile.TemporaryDirectory() as tmpdir:
            outpath = Path(tmpdir) / "metadata.json"
            save_pocket_metadata_json(pocket, str(outpath))
            payload = json.loads(outpath.read_text(encoding="utf-8"))

        self.assertIn("is_first_shell", payload["residues"][0])
        self.assertIn("is_second_shell", payload["residues"][0])


class GraphDatasetRuntimeTests(unittest.TestCase):
    def test_apply_feature_normalization_normalizes_and_clamps(self) -> None:
        pocket_a = make_pocket(
            [
                make_residue(chain_id="A", resseq=1, ca=(0.0, 0.0, 0.0), cb=(1.0, 0.0, 0.0)),
                make_residue(chain_id="A", resseq=2, ca=(0.0, 0.0, 2.0), cb=(1.0, 0.0, 2.0)),
            ]
        )
        pocket_b = make_pocket(
            [
                make_residue(chain_id="A", resseq=3, ca=(5.0, 0.0, 0.0), cb=(6.0, 0.0, 0.0)),
                make_residue(chain_id="A", resseq=4, ca=(5.0, 0.0, 2.0), cb=(6.0, 0.0, 2.0)),
            ]
        )
        for pocket in (pocket_a, pocket_b):
            for residue in pocket.residues:
                residue.esm_embedding = torch.ones(4, dtype=torch.float32)
                residue.has_esm_embedding = True
                residue.external_features = {"SASA": 1.0}
                residue.has_external_features = True

        data_a = pocket_to_pyg_data(pocket_a, esm_dim=4, edge_radius=3.0)
        data_b = pocket_to_pyg_data(pocket_b, esm_dim=4, edge_radius=3.0)
        stats = compute_feature_normalization_stats([data_a, data_b], clamp_value=1.0)
        normalized = apply_feature_normalization(data_b, stats)

        self.assertTrue(torch.all(torch.abs(normalized.x_dist_raw) <= 1.00001))
        self.assertTrue(torch.all(torch.abs(normalized.edge_dist_raw) <= 1.00001))


class StructureParsingTests(unittest.TestCase):
    def test_cluster_metal_records_merges_close_metal_sites(self) -> None:
        metal_records = [
            MetalAtomRecord(coord=torch.tensor([0.0, 0.0, 0.0]), symbol="ZN", site_id=("A", 100, "")),
            MetalAtomRecord(coord=torch.tensor([3.0, 0.0, 0.0]), symbol="CU", site_id=("A", 101, "")),
            MetalAtomRecord(coord=torch.tensor([10.0, 0.0, 0.0]), symbol="FE", site_id=("A", 102, "")),
        ]

        clusters = cluster_metal_records(metal_records, merge_distance=4.5)

        self.assertEqual([len(cluster) for cluster in clusters], [2, 1])


if __name__ == "__main__":
    unittest.main()
