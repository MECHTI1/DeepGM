from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from deepgm_colab import (
    apply_colab_defaults,
    cli_option_present,
    parse_colab_args,
    validate_colab_runtime_inputs,
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
        self.assertEqual(
            updated.summary_csv,
            Path(
                "/content/drive/MyDrive/DeepGM/"
                "train_set/data_summarizing_tables/final_data_summarazing_table_transition_metals_only_catalytic.csv"
            ),
        )
        self.assertEqual(updated.esm_embeddings_dir, "/content/drive/MyDrive/DeepGM/embeddings")
        self.assertEqual(updated.runs_dir, "/content/drive/MyDrive/DeepGM/training_runs")
        self.assertEqual(updated.device, "cpu")

    @patch("deepgm_colab.torch.cuda.is_available", return_value=True)
    def test_apply_colab_defaults_preserves_explicit_training_args(self, _mock_cuda_available) -> None:
        config = parse_args(
            [
                "--structure-dir",
                "/tmp/structures",
                "--summary-csv",
                "/tmp/summary.csv",
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
                "--summary-csv",
                "/tmp/summary.csv",
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
        self.assertEqual(updated.summary_csv, Path("/tmp/summary.csv"))
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
        self.assertEqual(
            config.summary_csv,
            Path(
                "/tmp/drive/DeepGM/"
                "train_set/data_summarizing_tables/final_data_summarazing_table_transition_metals_only_catalytic.csv"
            ),
        )


class ColabValidationTests(unittest.TestCase):
    def test_validate_colab_runtime_inputs_accepts_existing_structure_and_summary_paths(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            structure_dir = root / "train_set"
            summary_csv = structure_dir / "data_summarizing_tables" / "summary.csv"
            structure_dir.mkdir(parents=True)
            summary_csv.parent.mkdir(parents=True, exist_ok=True)
            summary_csv.write_text("pdbid\n", encoding="utf-8")

            config = parse_args(
                [
                    "--structure-dir",
                    str(structure_dir),
                    "--summary-csv",
                    str(summary_csv),
                    "--esm-embeddings-dir",
                    str(root / "embeddings"),
                    "--runs-dir",
                    str(root / "runs"),
                ]
            )

            validate_colab_runtime_inputs(config)

            self.assertTrue((root / "embeddings").is_dir())
            self.assertTrue((root / "runs").is_dir())

    def test_validate_colab_runtime_inputs_rejects_missing_summary_csv(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            structure_dir = root / "train_set"
            structure_dir.mkdir(parents=True)
            config = parse_args(
                [
                    "--structure-dir",
                    str(structure_dir),
                    "--summary-csv",
                    str(root / "missing.csv"),
                ]
            )

            with self.assertRaises(FileNotFoundError):
                validate_colab_runtime_inputs(config)


if __name__ == "__main__":
    unittest.main()
