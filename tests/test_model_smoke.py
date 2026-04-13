from __future__ import annotations

import unittest

import torch
from torch_geometric.loader import DataLoader

from data_structures import PocketRecord, ResidueRecord
from label_schemes import N_EC_CLASSES, N_METAL_CLASSES
from model import GVPPocketClassifier
from training.graph_dataset import (
    PocketGraphDataset,
    build_graph_data_list,
    compute_feature_normalization_stats,
)


def make_residue(
    *,
    chain_id: str,
    resseq: int,
    ca: tuple[float, float, float],
    cb: tuple[float, float, float],
    nd1: tuple[float, float, float],
    esm_dim: int,
) -> ResidueRecord:
    return ResidueRecord(
        chain_id=chain_id,
        resseq=resseq,
        icode="",
        resname="HIS",
        atoms={
            "CA": torch.tensor(ca, dtype=torch.float32),
            "CB": torch.tensor(cb, dtype=torch.float32),
            "ND1": torch.tensor(nd1, dtype=torch.float32),
        },
        esm_embedding=torch.linspace(0.1, 0.1 * esm_dim, steps=esm_dim, dtype=torch.float32),
        has_esm_embedding=True,
        is_first_shell=True,
        external_features={
            "SASA": 1.0,
            "BSA": 0.5,
            "SolvEnergy": -0.2,
            "fa_sol": 0.3,
            "pKa_shift": 0.1,
            "dpKa_desolv": 0.2,
            "dpKa_bg": -0.1,
            "dpKa_titr": 0.05,
            "omega": 0.6,
            "rama_prepro": -0.4,
            "fa_dun": 0.8,
            "fa_elec": -0.3,
            "fa_atr": 0.7,
            "fa_rep": 0.2,
        },
        has_external_features=True,
    )


def make_pocket(*, pocket_id: str, esm_dim: int, metal_shift: float, y_metal: int, y_ec: int) -> PocketRecord:
    residues = [
        make_residue(
            chain_id="A",
            resseq=10,
            ca=(metal_shift + 0.0, 0.0, 0.0),
            cb=(metal_shift + 0.8, 0.5, 0.0),
            nd1=(metal_shift + 0.4, 0.2, 0.0),
            esm_dim=esm_dim,
        ),
        make_residue(
            chain_id="A",
            resseq=11,
            ca=(metal_shift + 0.0, 0.0, 2.0),
            cb=(metal_shift + 0.7, 0.4, 2.1),
            nd1=(metal_shift + 0.2, 0.1, 1.5),
            esm_dim=esm_dim,
        ),
    ]
    return PocketRecord(
        structure_id=f"{pocket_id}__chain_A__EC_1.1.1.1",
        pocket_id=pocket_id,
        metal_element="ZN",
        metal_coords=[torch.tensor([metal_shift, 0.0, 0.8], dtype=torch.float32)],
        residues=residues,
        y_metal=y_metal,
        y_ec=y_ec,
    )


class ModelSmokeTests(unittest.TestCase):
    def test_precomputed_graphs_are_cloned_before_normalization(self) -> None:
        esm_dim = 8
        pockets = [
            make_pocket(pocket_id="pocket-a", esm_dim=esm_dim, metal_shift=0.0, y_metal=0, y_ec=1),
        ]
        precomputed = build_graph_data_list(pockets, esm_dim=esm_dim, edge_radius=3.0)
        original_x_misc = precomputed[0].x_misc.clone()
        normalization_stats = compute_feature_normalization_stats(precomputed)

        dataset = PocketGraphDataset(
            pockets,
            esm_dim=esm_dim,
            edge_radius=3.0,
            normalization_stats=normalization_stats,
            precomputed_data=precomputed,
        )
        first = dataset[0]
        second = dataset[0]

        self.assertFalse(torch.equal(first.x_misc, original_x_misc))
        self.assertTrue(torch.equal(precomputed[0].x_misc, original_x_misc))
        self.assertTrue(torch.equal(first.x_misc, second.x_misc))

    def test_graph_dataset_item_batches_into_model_forward(self) -> None:
        torch.manual_seed(0)
        esm_dim = 8
        pockets = [
            make_pocket(pocket_id="pocket-a", esm_dim=esm_dim, metal_shift=0.0, y_metal=0, y_ec=1),
            make_pocket(pocket_id="pocket-b", esm_dim=esm_dim, metal_shift=5.0, y_metal=1, y_ec=2),
        ]

        dataset = PocketGraphDataset(pockets, esm_dim=esm_dim, edge_radius=3.0)
        loader = DataLoader(dataset, batch_size=2, shuffle=False)
        batch = next(iter(loader))

        model = GVPPocketClassifier(esm_dim=esm_dim)
        model.eval()
        outputs = model(batch)

        self.assertIn("logits_metal", outputs)
        self.assertIn("logits_ec", outputs)
        self.assertIn("loss", outputs)
        self.assertEqual(tuple(outputs["logits_metal"].shape), (2, N_METAL_CLASSES))
        self.assertEqual(tuple(outputs["logits_ec"].shape), (2, N_EC_CLASSES))
        self.assertEqual(tuple(outputs["embed"].shape), (2, 288))
        self.assertTrue(torch.isfinite(outputs["loss"]))


if __name__ == "__main__":
    unittest.main()
