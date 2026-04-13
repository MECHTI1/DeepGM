from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from data_structures import PocketRecord, ResidueRecord
from training.config import parse_args
from training.graph_dataset import FeatureNormalizationStats
from training.run import (
    PocketSplit,
    PreparedRun,
    build_dataset_summary,
    split_pockets,
    train_and_select_checkpoint,
    validate_training_configuration,
)


def make_residue(
    *,
    chain_id: str = "A",
    resseq: int = 1,
    resname: str = "HIS",
    has_esm_embedding: bool = False,
    has_external_features: bool = False,
) -> ResidueRecord:
    return ResidueRecord(
        chain_id=chain_id,
        resseq=resseq,
        icode="",
        resname=resname,
        atoms={"CA": torch.tensor([0.0, 0.0, 0.0])},
        has_esm_embedding=has_esm_embedding,
        has_external_features=has_external_features,
    )


def make_pocket(
    *,
    structure_id: str,
    pocket_id: str,
    residue: ResidueRecord | None = None,
    y_metal: int | None = None,
    y_ec: int | None = None,
) -> PocketRecord:
    return PocketRecord(
        structure_id=structure_id,
        pocket_id=pocket_id,
        metal_element="ZN",
        metal_coords=[torch.tensor([1.0, 2.0, 3.0])],
        residues=[residue or make_residue()],
        y_metal=y_metal,
        y_ec=y_ec,
    )


class TrainConfigParsingTests(unittest.TestCase):
    def test_parse_args_defaults_to_train_loss_without_validation(self) -> None:
        config = parse_args([])

        self.assertEqual(config.val_fraction, 0.0)
        self.assertEqual(config.selection_metric, "train_loss")

    def test_parse_args_builds_expected_config(self) -> None:
        config = parse_args(
            [
                "--structure-dir",
                "/tmp/structures",
                "--summary-csv",
                "/tmp/summary.csv",
                "--epochs",
                "3",
                "--batch-size",
                "16",
                "--split-by",
                "structure_id",
                "--allow-missing-esm-embeddings",
                "--unsupported-metal-policy",
                "skip",
                "--selection-metric",
                "val_loss",
            ]
        )

        self.assertEqual(config.structure_dir, Path("/tmp/structures"))
        self.assertEqual(config.summary_csv, Path("/tmp/summary.csv"))
        self.assertEqual(config.epochs, 3)
        self.assertEqual(config.batch_size, 16)
        self.assertEqual(config.split_by, "structure_id")
        self.assertFalse(config.require_esm_embeddings)
        self.assertTrue(config.require_external_features)
        self.assertEqual(config.unsupported_metal_policy, "skip")
        self.assertEqual(config.selection_metric, "val_loss")


class TrainingSplitTests(unittest.TestCase):
    def test_split_pockets_keeps_same_pdbid_in_one_split(self) -> None:
        pockets = [
            make_pocket(structure_id="1abc__chain_A__EC_1.1.1.1", pocket_id="p1"),
            make_pocket(structure_id="1abc__chain_B__EC_2.2.2.2", pocket_id="p2"),
            make_pocket(structure_id="2def__chain_A__EC_3.3.3.3", pocket_id="p3"),
            make_pocket(structure_id="3ghi__chain_A__EC_4.4.4.4", pocket_id="p4"),
        ]

        split = split_pockets(pockets, val_fraction=0.5, split_by="pdbid", seed=7)

        train_pdbids = {pocket.structure_id.split("__", 1)[0] for pocket in split.train_pockets}
        val_pdbids = {pocket.structure_id.split("__", 1)[0] for pocket in split.val_pockets}
        self.assertFalse(train_pdbids & val_pdbids)
        self.assertEqual(
            sorted(pocket.pocket_id for pocket in pockets),
            sorted(pocket.pocket_id for pocket in split.train_pockets + split.val_pockets),
        )

    def test_zero_val_fraction_returns_all_training_pockets(self) -> None:
        pockets = [make_pocket(structure_id="1abc__chain_A__EC_1.1.1.1", pocket_id="p1")]

        split = split_pockets(pockets, val_fraction=0.0, split_by="pdbid", seed=1)

        self.assertEqual(split.train_pockets, pockets)
        self.assertEqual(split.val_pockets, [])


class DatasetSummaryTests(unittest.TestCase):
    def test_build_dataset_summary_reports_split_counts_and_label_distribution(self) -> None:
        train_pocket = make_pocket(
            structure_id="1abc__chain_A__EC_1.1.1.1",
            pocket_id="train-pocket",
            residue=make_residue(has_esm_embedding=True, has_external_features=False),
            y_metal=0,
            y_ec=1,
        )
        val_pocket = make_pocket(
            structure_id="2def__chain_A__EC_2.2.2.2",
            pocket_id="val-pocket",
            residue=make_residue(has_esm_embedding=False, has_external_features=True),
            y_metal=2,
            y_ec=4,
        )
        split = PocketSplit(train_pockets=[train_pocket], val_pockets=[val_pocket])
        config = parse_args(
            [
                "--structure-dir",
                "/tmp/structures",
                "--summary-csv",
                "/tmp/summary.csv",
                "--val-fraction",
                "0.25",
            ]
        )

        summary = build_dataset_summary(
            split,
            config,
            feature_load_report={"feature_fallbacks": [], "loaded_structure_files": 2},
        )

        self.assertEqual(summary["n_train_pockets"], 1)
        self.assertEqual(summary["n_val_pockets"], 1)
        self.assertEqual(summary["split_by"], "pdbid")
        self.assertEqual(summary["selection_metric"], "val_joint_balanced_acc")
        self.assertEqual(summary["unsupported_metal_policy"], "error")
        self.assertEqual(summary["train_metal_distribution"]["Mn"], 1)
        self.assertEqual(summary["val_metal_distribution"]["Zn"], 1)
        self.assertEqual(summary["train_feature_coverage"]["esm_residue_coverage"], 1.0)
        self.assertEqual(summary["val_feature_coverage"]["external_feature_residue_coverage"], 1.0)


class TrainingLoopHistoryTests(unittest.TestCase):
    @patch("training.run.train_epoch", return_value=0.75)
    @patch("training.run.evaluate_split_metrics")
    def test_train_and_select_checkpoint_uses_train_loss_without_validation(
        self,
        mock_evaluate_split_metrics,
        _mock_train_epoch,
    ) -> None:
        mock_evaluate_split_metrics.side_effect = [
            {
                "train_loss": 0.11,
                "train_metal_acc": 0.8,
                "train_ec_acc": 0.6,
            },
            {},
        ]

        model = torch.nn.Linear(1, 1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        prepared = PreparedRun(
            config_payload={},
            run_dir=Path("/tmp"),
            split=PocketSplit(train_pockets=[], val_pockets=[]),
            dataset_summary={},
            normalization_stats=FeatureNormalizationStats(means={}, stds={}),
            train_loader=None,
            val_loader=None,
            model=model,
            optimizer=optimizer,
        )
        config = parse_args(["--epochs", "1"])

        history, best_checkpoint = train_and_select_checkpoint(prepared, config)

        self.assertEqual(history[0]["train_loss"], 0.75)
        self.assertIsNotNone(best_checkpoint)
        self.assertEqual(best_checkpoint["epoch"], 1)
        self.assertEqual(best_checkpoint["selection_metric"], "train_loss")
        self.assertEqual(best_checkpoint["selection_metric_value"], 0.75)

    @patch("training.run.train_epoch", return_value=0.75)
    @patch("training.run.evaluate_split_metrics")
    def test_train_and_select_checkpoint_keeps_optimizer_train_loss(
        self,
        mock_evaluate_split_metrics,
        _mock_train_epoch,
    ) -> None:
        def metrics_side_effect(model, loader, device, prefix):
            if prefix == "train":
                return {
                    "train_loss": 0.11,
                    "train_metal_acc": 0.8,
                    "train_ec_acc": 0.6,
                    "train_joint_balanced_acc": 0.7,
                }
            return {
                "val_loss": 0.4,
                "val_metal_acc": 0.7,
                "val_ec_acc": 0.5,
                "val_joint_balanced_acc": 0.65,
            }

        mock_evaluate_split_metrics.side_effect = metrics_side_effect

        model = torch.nn.Linear(1, 1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        prepared = PreparedRun(
            config_payload={},
            run_dir=Path("/tmp"),
            split=PocketSplit(train_pockets=[], val_pockets=[]),
            dataset_summary={},
            normalization_stats=FeatureNormalizationStats(means={}, stds={}),
            train_loader=None,
            val_loader=None,
            model=model,
            optimizer=optimizer,
        )
        config = parse_args(["--epochs", "1", "--selection-metric", "val_joint_balanced_acc"])

        history, best_checkpoint = train_and_select_checkpoint(prepared, config)

        self.assertEqual(history[0]["train_loss"], 0.75)
        self.assertEqual(history[0]["train_metal_acc"], 0.8)
        self.assertEqual(history[0]["train_ec_acc"], 0.6)
        self.assertIsNotNone(best_checkpoint)
        self.assertEqual(best_checkpoint["epoch"], 1)
        self.assertEqual(best_checkpoint["selection_metric"], "val_joint_balanced_acc")
        self.assertEqual(best_checkpoint["selection_metric_value"], 0.65)


class TrainingConfigurationValidationTests(unittest.TestCase):
    def test_validation_metric_requires_validation_split(self) -> None:
        config = parse_args(["--selection-metric", "val_joint_balanced_acc"])

        with self.assertRaisesRegex(ValueError, "requires validation"):
            validate_training_configuration(config)


if __name__ == "__main__":
    unittest.main()
