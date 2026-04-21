from __future__ import annotations

import unittest

import torch

from data_structures import PocketRecord, ResidueRecord
from training.config import parse_args
from training.preflight import run_preflight_checks
from training.splits import PocketSplit


def make_residue(
    *,
    chain_id: str,
    resseq: int,
    ca: tuple[float, float, float],
    cb: tuple[float, float, float],
    nd1: tuple[float, float, float],
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
        has_esm_embedding=False,
        has_external_features=False,
        is_first_shell=True,
    )


def make_graphable_pocket(*, pocket_id: str, shift: float, y_metal: int, y_ec: int) -> PocketRecord:
    residues = [
        make_residue(
            chain_id="A",
            resseq=1,
            ca=(shift + 0.0, 0.0, 0.0),
            cb=(shift + 0.8, 0.5, 0.0),
            nd1=(shift + 0.4, 0.2, 0.0),
        ),
        make_residue(
            chain_id="A",
            resseq=2,
            ca=(shift + 0.0, 0.0, 2.0),
            cb=(shift + 0.7, 0.4, 2.1),
            nd1=(shift + 0.2, 0.1, 1.5),
        ),
    ]
    return PocketRecord(
        structure_id=f"{pocket_id}__chain_A__EC_1.1.1.1",
        pocket_id=pocket_id,
        metal_element="ZN",
        metal_coords=[torch.tensor([shift, 0.0, 0.8], dtype=torch.float32)],
        residues=residues,
        y_metal=y_metal,
        y_ec=y_ec,
    )


class TrainingPreflightTests(unittest.TestCase):
    def test_run_preflight_checks_builds_report(self) -> None:
        split = PocketSplit(
            train_pockets=[
                make_graphable_pocket(pocket_id="train-a", shift=0.0, y_metal=0, y_ec=1),
                make_graphable_pocket(pocket_id="train-b", shift=4.0, y_metal=1, y_ec=2),
            ],
            val_pockets=[],
        )
        config = parse_args(["--esm-dim", "8", "--edge-radius", "3.0"])

        report = run_preflight_checks(split, config)

        self.assertEqual(
            report,
            {
                "warnings": [
                    "Training split has no ESM residue coverage.",
                    "Training split has no external feature residue coverage.",
                ]
            },
        )

    def test_run_preflight_checks_rejects_pocket_without_residues(self) -> None:
        split = PocketSplit(
            train_pockets=[
                PocketRecord(
                    structure_id="bad__chain_A__EC_1.1.1.1",
                    pocket_id="bad-pocket",
                    metal_element="ZN",
                    metal_coords=[torch.tensor([0.0, 0.0, 0.0])],
                    residues=[],
                    y_metal=0,
                    y_ec=1,
                )
            ],
            val_pockets=[],
        )
        config = parse_args([])

        with self.assertRaisesRegex(ValueError, "training pockets without residues"):
            run_preflight_checks(split, config)

    def test_run_preflight_checks_rejects_non_graphable_validation_pocket(self) -> None:
        split = PocketSplit(
            train_pockets=[
                make_graphable_pocket(pocket_id="train-a", shift=0.0, y_metal=0, y_ec=1),
                make_graphable_pocket(pocket_id="train-b", shift=4.0, y_metal=1, y_ec=2),
            ],
            val_pockets=[
                make_graphable_pocket(pocket_id="val-bad", shift=100.0, y_metal=0, y_ec=1),
            ],
        )
        split.val_pockets[0].residues[1].atoms["CA"] = torch.tensor([200.0, 0.0, 0.0], dtype=torch.float32)
        split.val_pockets[0].residues[1].atoms["CB"] = torch.tensor([200.7, 0.4, 2.1], dtype=torch.float32)
        split.val_pockets[0].residues[1].atoms["ND1"] = torch.tensor([200.2, 0.1, 1.5], dtype=torch.float32)
        config = parse_args(["--esm-dim", "8", "--edge-radius", "3.0", "--val-fraction", "0.25"])

        with self.assertRaisesRegex(ValueError, "Graph preflight failed for pocket 'val-bad'"):
            run_preflight_checks(split, config)

    def test_run_preflight_checks_rejects_late_non_graphable_training_pocket(self) -> None:
        split = PocketSplit(
            train_pockets=[
                make_graphable_pocket(pocket_id="train-a", shift=0.0, y_metal=0, y_ec=1),
                make_graphable_pocket(pocket_id="train-b", shift=4.0, y_metal=1, y_ec=2),
                make_graphable_pocket(pocket_id="train-c", shift=8.0, y_metal=0, y_ec=1),
                make_graphable_pocket(pocket_id="train-bad", shift=100.0, y_metal=1, y_ec=2),
            ],
            val_pockets=[],
        )
        split.train_pockets[3].residues[1].atoms["CA"] = torch.tensor([200.0, 0.0, 0.0], dtype=torch.float32)
        split.train_pockets[3].residues[1].atoms["CB"] = torch.tensor([200.7, 0.4, 2.1], dtype=torch.float32)
        split.train_pockets[3].residues[1].atoms["ND1"] = torch.tensor([200.2, 0.1, 1.5], dtype=torch.float32)
        config = parse_args(["--esm-dim", "8", "--edge-radius", "3.0"])

        with self.assertRaisesRegex(ValueError, "Graph preflight failed for pocket 'train-bad'"):
            run_preflight_checks(split, config)

    def test_run_preflight_checks_rejects_split_leakage(self) -> None:
        split = PocketSplit(
            train_pockets=[
                make_graphable_pocket(pocket_id="train-a", shift=0.0, y_metal=0, y_ec=1),
                make_graphable_pocket(pocket_id="train-b", shift=4.0, y_metal=1, y_ec=2),
            ],
            val_pockets=[
                make_graphable_pocket(pocket_id="train-a-other", shift=4.0, y_metal=1, y_ec=2),
            ],
        )
        split.val_pockets[0].structure_id = split.train_pockets[0].structure_id
        config = parse_args(["--esm-dim", "8", "--edge-radius", "3.0", "--val-fraction", "0.25"])

        with self.assertRaisesRegex(ValueError, "train/validation leakage detected"):
            run_preflight_checks(split, config)

    def test_run_preflight_checks_rejects_single_class_training_target(self) -> None:
        split = PocketSplit(
            train_pockets=[
                make_graphable_pocket(pocket_id="train-a", shift=0.0, y_metal=0, y_ec=1),
                make_graphable_pocket(pocket_id="train-b", shift=4.0, y_metal=0, y_ec=1),
            ],
            val_pockets=[],
        )
        config = parse_args(["--esm-dim", "8", "--edge-radius", "3.0"])

        with self.assertRaisesRegex(ValueError, "fewer than 2 metal classes"):
            run_preflight_checks(split, config)

    def test_run_preflight_checks_allows_single_metal_class_for_ec_task(self) -> None:
        split = PocketSplit(
            train_pockets=[
                make_graphable_pocket(pocket_id="train-a", shift=0.0, y_metal=0, y_ec=1),
                make_graphable_pocket(pocket_id="train-b", shift=4.0, y_metal=0, y_ec=2),
            ],
            val_pockets=[],
        )
        config = parse_args(["--task", "ec", "--esm-dim", "8", "--edge-radius", "3.0"])

        report = run_preflight_checks(split, config)

        self.assertEqual(
            report,
            {
                "warnings": [
                    "Training split has no ESM residue coverage.",
                    "Training split has no external feature residue coverage.",
                ]
            },
        )


if __name__ == "__main__":
    unittest.main()
