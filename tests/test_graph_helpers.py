from __future__ import annotations

import unittest

import torch

from data_structures import PocketRecord, ResidueRecord
from graph.edge_building import build_pair_edge_geometry, build_radius_graph_from_residues, stack_edge_features
from graph.feature_utils import attach_esm_embeddings, attach_external_residue_features


def make_residue(
    *,
    chain_id: str,
    resseq: int,
    ca: tuple[float, float, float],
    cb: tuple[float, float, float] | None = None,
) -> ResidueRecord:
    atoms = {"CA": torch.tensor(ca, dtype=torch.float32)}
    if cb is not None:
        atoms["CB"] = torch.tensor(cb, dtype=torch.float32)
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

    def test_build_radius_graph_from_residues_returns_directed_pairs_within_cutoff(self) -> None:
        residues = [
            make_residue(chain_id="A", resseq=1, ca=(0.0, 0.0, 0.0)),
            make_residue(chain_id="A", resseq=2, ca=(0.0, 0.0, 2.0)),
            make_residue(chain_id="A", resseq=3, ca=(0.0, 0.0, 6.0)),
        ]

        edge_index = build_radius_graph_from_residues(residues, radius=2.5)

        self.assertEqual(edge_index.tolist(), [[0, 1], [1, 0]])

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
            ]
        )

        self.assertEqual(edge_features["edge_index"].tolist(), [[0], [1]])
        self.assertEqual(tuple(edge_features["edge_dist_raw"].shape), (1, 2))
        self.assertEqual(tuple(edge_features["edge_seqsep"].shape), (1, 1))
        self.assertEqual(tuple(edge_features["edge_source_type"].shape), (1, 2))


if __name__ == "__main__":
    unittest.main()
