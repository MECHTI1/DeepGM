from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from training.runtime_preparation import discover_missing_esm_embeddings, prepare_runtime_inputs


class RuntimePreparationTests(unittest.TestCase):
    def test_discover_missing_esm_embeddings_skips_existing_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            structure_a = root / "1abc__chain_A__EC_1.1.1.1.pdb"
            structure_b = root / "2def__chain_A__EC_2.2.2.2.pdb"
            structure_a.write_text("ATOM\n", encoding="utf-8")
            structure_b.write_text("ATOM\n", encoding="utf-8")

            embeddings_dir = root / "embeddings"
            embeddings_dir.mkdir()
            (embeddings_dir / "1abc__chain_A__EC_1.1.1.1_chain_A_esmc.pt").write_text("", encoding="utf-8")

            missing = discover_missing_esm_embeddings([structure_a, structure_b], embeddings_dir)

            self.assertEqual(missing, [structure_b])

    @patch("training.runtime_preparation._generate_missing_esm_embeddings")
    @patch("training.runtime_preparation.find_structure_files")
    def test_prepare_runtime_inputs_generates_only_missing_esm_files(
        self,
        mock_find_structure_files,
        mock_generate_missing_esm_embeddings,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            structure_a = root / "1abc__chain_A__EC_1.1.1.1.pdb"
            structure_b = root / "2def__chain_A__EC_2.2.2.2.pdb"
            structure_a.write_text("ATOM\n", encoding="utf-8")
            structure_b.write_text("ATOM\n", encoding="utf-8")
            mock_find_structure_files.return_value = [structure_a, structure_b]

            embeddings_dir = root / "embeddings"
            embeddings_dir.mkdir()
            (embeddings_dir / "1abc__chain_A__EC_1.1.1.1_chain_A_esmc.pt").write_text("", encoding="utf-8")
            mock_generate_missing_esm_embeddings.return_value = {
                "saved_files": [str(embeddings_dir / "2def__chain_A__EC_2.2.2.2_chain_A_esmc.pt")],
                "failed_structures": [],
            }

            report = prepare_runtime_inputs(
                structure_dir=root,
                esm_embeddings_dir=embeddings_dir,
                require_esm_embeddings=True,
                prepare_missing_esm_embeddings=True,
                require_ring_edges=False,
                prepare_missing_ring_edges=False,
            )

            mock_generate_missing_esm_embeddings.assert_called_once_with([structure_b], embeddings_dir)
            self.assertEqual(report["missing_esm_structures_before"], 1)
            self.assertEqual(report["generated_esm_files"], 1)
            self.assertEqual(report["generated_ring_edge_files"], 0)

    @patch("training.runtime_preparation._generate_missing_updated_external_features")
    @patch("training.runtime_preparation.find_structure_files")
    def test_prepare_runtime_inputs_generates_missing_updated_external_features(
        self,
        mock_find_structure_files,
        mock_generate_missing_updated_external_features,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            structure = root / "1abc__chain_A__EC_1.1.1.1.pdb"
            structure.write_text("ATOM\n", encoding="utf-8")
            mock_find_structure_files.return_value = [structure]

            updated_root = root / "updated_features"
            mock_generate_missing_updated_external_features.return_value = {
                "saved_files": [str(updated_root / structure.stem / "residue_features.json")],
                "failed_structures": [],
            }

            report = prepare_runtime_inputs(
                structure_dir=root,
                esm_embeddings_dir=root / "embeddings",
                require_esm_embeddings=False,
                prepare_missing_esm_embeddings=False,
                require_ring_edges=False,
                prepare_missing_ring_edges=False,
                external_features_root_dir=updated_root,
                external_feature_source="updated",
                require_external_features=True,
            )

            mock_generate_missing_updated_external_features.assert_called_once_with([structure], updated_root)
            self.assertEqual(report["missing_updated_external_feature_structures_before"], 1)
            self.assertEqual(report["generated_updated_external_feature_files"], 1)

    @patch("training.runtime_preparation.find_structure_files")
    def test_prepare_runtime_inputs_skips_esm_generation_when_disabled(self, mock_find_structure_files) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            structure = root / "1abc__chain_A__EC_1.1.1.1.pdb"
            structure.write_text("ATOM\n", encoding="utf-8")
            mock_find_structure_files.return_value = [structure]

            report = prepare_runtime_inputs(
                structure_dir=root,
                esm_embeddings_dir=root / "embeddings",
                require_esm_embeddings=True,
                prepare_missing_esm_embeddings=False,
                require_ring_edges=False,
                prepare_missing_ring_edges=False,
            )

            self.assertEqual(report["missing_esm_structures_before"], 0)
            self.assertEqual(report["generated_esm_files"], 0)
            self.assertEqual(report["generated_updated_external_feature_files"], 0)

    @patch("training.runtime_preparation._generate_missing_esm_embeddings")
    @patch("training.runtime_preparation.find_structure_files")
    def test_prepare_runtime_inputs_raises_on_generation_failure(
        self,
        mock_find_structure_files,
        mock_generate_missing_esm_embeddings,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            structure = root / "1abc__chain_A__EC_1.1.1.1.pdb"
            structure.write_text("ATOM\n", encoding="utf-8")
            mock_find_structure_files.return_value = [structure]
            mock_generate_missing_esm_embeddings.return_value = {
                "saved_files": [],
                "failed_structures": [{"structure_file": str(structure), "error": "boom"}],
            }

            with self.assertRaisesRegex(ValueError, "Failed to generate missing ESM embeddings"):
                prepare_runtime_inputs(
                    structure_dir=root,
                    esm_embeddings_dir=root / "embeddings",
                    require_esm_embeddings=True,
                    prepare_missing_esm_embeddings=True,
                    require_ring_edges=False,
                    prepare_missing_ring_edges=False,
                )

    @patch("training.runtime_preparation._generate_missing_updated_external_features")
    @patch("training.runtime_preparation.find_structure_files")
    def test_prepare_runtime_inputs_raises_on_updated_external_feature_generation_failure(
        self,
        mock_find_structure_files,
        mock_generate_missing_updated_external_features,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            structure = root / "1abc__chain_A__EC_1.1.1.1.pdb"
            structure.write_text("ATOM\n", encoding="utf-8")
            mock_find_structure_files.return_value = [structure]
            mock_generate_missing_updated_external_features.return_value = {
                "saved_files": [],
                "failed_structures": [{"structure_path": str(structure), "error": "boom"}],
            }

            with self.assertRaisesRegex(ValueError, "Failed to generate missing updated external features"):
                prepare_runtime_inputs(
                    structure_dir=root,
                    esm_embeddings_dir=root / "embeddings",
                    require_esm_embeddings=False,
                    prepare_missing_esm_embeddings=False,
                    require_ring_edges=False,
                    prepare_missing_ring_edges=False,
                    external_features_root_dir=root / "updated_features",
                    external_feature_source="updated",
                    require_external_features=True,
                )


if __name__ == "__main__":
    unittest.main()
