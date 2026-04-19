from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

from deepgm_colab import (
    apply_colab_defaults,
    cli_option_present,
    parse_colab_args,
)
from training.config import parse_args


class ColabCliHelpersTests(unittest.TestCase):
    def test_cli_option_present_accepts_split_or_equals_form(self) -> None:
        argv = ["--structure-dir", "/tmp/structures", "--runs-dir=/tmp/runs"]

        self.assertTrue(cli_option_present(argv, "--structure-dir"))
        self.assertTrue(cli_option_present(argv, "--runs-dir"))
        self.assertFalse(cli_option_present(argv, "--device"))


class ColabDefaultingTests(unittest.TestCase):
    @patch("deepgm_colab.torch.cuda.is_available", return_value=False)
    def test_apply_colab_defaults_overrides_missing_paths_and_device(self, _mock_cuda_available) -> None:
        config = parse_args([])

        updated = apply_colab_defaults(
            config,
            argv=[],
            drive_root=Path("/content/drive/MyDrive"),
        )

        self.assertEqual(updated.structure_dir, Path("/content/drive/MyDrive/DeepGM/train_set"))
        self.assertEqual(updated.esm_embeddings_dir, "/content/drive/MyDrive/DeepGM/embeddings")
        self.assertEqual(updated.runs_dir, "/content/drive/MyDrive/DeepGM/training_runs")
        self.assertEqual(updated.device, "cpu")

    @patch("deepgm_colab.torch.cuda.is_available", return_value=True)
    def test_apply_colab_defaults_preserves_explicit_training_args(self, _mock_cuda_available) -> None:
        config = parse_args(
            [
                "--structure-dir",
                "/tmp/structures",
                "--esm-embeddings-dir",
                "/tmp/embeddings",
                "--runs-dir",
                "/tmp/runs",
                "--device",
                "cpu",
            ]
        )

        updated = apply_colab_defaults(
            config,
            argv=[
                "--structure-dir",
                "/tmp/structures",
                "--esm-embeddings-dir",
                "/tmp/embeddings",
                "--runs-dir",
                "/tmp/runs",
                "--device",
                "cpu",
            ],
            drive_root=Path("/content/drive/MyDrive"),
        )

        self.assertEqual(updated.structure_dir, Path("/tmp/structures"))
        self.assertEqual(updated.esm_embeddings_dir, "/tmp/embeddings")
        self.assertEqual(updated.runs_dir, "/tmp/runs")
        self.assertEqual(updated.device, "cpu")


class ColabParsingTests(unittest.TestCase):
    @patch("deepgm_colab.torch.cuda.is_available", return_value=False)
    def test_parse_colab_args_splits_wrapper_and_training_args(self, _mock_cuda_available) -> None:
        wrapper_args, config, training_argv = parse_colab_args(
            [
                "--mount-drive",
                "--drive-root",
                "/tmp/drive",
                "--epochs",
                "3",
            ]
        )

        self.assertTrue(wrapper_args.mount_drive)
        self.assertEqual(wrapper_args.drive_root, Path("/tmp/drive"))
        self.assertEqual(training_argv, ["--epochs", "3"])
        self.assertEqual(config.epochs, 3)
        self.assertEqual(config.structure_dir, Path("/tmp/drive/DeepGM/train_set"))


if __name__ == "__main__":
    unittest.main()
