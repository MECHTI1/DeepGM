from __future__ import annotations

import unittest

from training.task_entrypoint import parse_separate_task_args


class SeparateTaskArgParsingTests(unittest.TestCase):
    def test_metal_defaults_are_strict_and_balanced(self) -> None:
        config = parse_separate_task_args("metal", [])

        self.assertEqual(config.task, "metal")
        self.assertEqual(config.external_feature_source, "updated")
        self.assertTrue(config.require_esm_embeddings)
        self.assertTrue(config.require_external_features)
        self.assertEqual(config.val_fraction, 0.2)
        self.assertEqual(config.batch_size, 8)
        self.assertEqual(config.selection_metric, "val_metal_balanced_acc")

    def test_ec_wrapper_allows_batch_size_sixteen(self) -> None:
        config = parse_separate_task_args("ec", ["--batch-size", "16"])

        self.assertEqual(config.task, "ec")
        self.assertEqual(config.batch_size, 16)
        self.assertEqual(config.selection_metric, "val_ec_balanced_acc")

    def test_wrapper_rejects_joint_task_override(self) -> None:
        with self.assertRaisesRegex(ValueError, "only supports --task"):
            parse_separate_task_args("metal", ["--task", "joint"])

    def test_wrapper_rejects_missing_esm_override(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires ESM embeddings"):
            parse_separate_task_args("metal", ["--allow-missing-esm-embeddings"])

    def test_wrapper_rejects_non_updated_external_features(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires --external-feature-source updated"):
            parse_separate_task_args("ec", ["--external-feature-source", "auto"])

    def test_wrapper_rejects_non_balanced_selection_metric(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires --selection-metric"):
            parse_separate_task_args("ec", ["--selection-metric", "train_loss"])

    def test_wrapper_rejects_batch_size_outside_supported_choices(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires --batch-size"):
            parse_separate_task_args("metal", ["--batch-size", "4"])


if __name__ == "__main__":
    unittest.main()
