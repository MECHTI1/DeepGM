from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from data_structures import PocketRecord, ResidueRecord
from graph.construction import parse_structure_file
from training.data import load_training_pockets_with_report_from_dir
from training.esm_feature_loading import (
    build_embedding_payload,
    load_esm_lookup_for_structure,
    residue_keys_for_structure_chain,
)
from training.external_feature_loading import structure_dir_to_feature_lookup
from training.structure_loading import StructureLoadError


SAMPLE_STRUCTURE_ID = "1a0e__chain_A__EC_5.3.1.5"
SAMPLE_JOB_DIR = Path("/media/Data/pinmymetal_sets/mahomes/train_set/job_0")
SAMPLE_FEATURE_DIR = SAMPLE_JOB_DIR / SAMPLE_STRUCTURE_ID
SAMPLE_STRUCTURE_PATH = SAMPLE_FEATURE_DIR / f"{SAMPLE_STRUCTURE_ID}.pdb"


def require_sample_data() -> None:
    if not SAMPLE_STRUCTURE_PATH.is_file():
        raise unittest.SkipTest(f"Missing MAHOMES sample structure: {SAMPLE_STRUCTURE_PATH}")


class TrainingDataLoadingTests(unittest.TestCase):
    def setUp(self) -> None:
        require_sample_data()
        self.tempdir = tempfile.TemporaryDirectory()
        self.workdir = Path(self.tempdir.name)
        self.structure_root = self.workdir / "structures"
        self.structure_root.mkdir()
        self.structure_path = self.structure_root / SAMPLE_STRUCTURE_PATH.name
        shutil.copy2(SAMPLE_STRUCTURE_PATH, self.structure_path)
        self.embeddings_dir = self.workdir / "embeddings"
        self.embeddings_dir.mkdir()

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def build_sample_embedding_file(self, esm_dim: int = 8, *, canonical: bool = True) -> Path:
        structure = parse_structure_file(str(self.structure_path), structure_id=self.structure_path.stem)
        residue_keys, residue_keys_with_ca = residue_keys_for_structure_chain(structure, "A")
        self.assertGreater(len(residue_keys), 0)
        self.assertGreater(len(residue_keys_with_ca), 0)

        embedding = torch.arange(
            len(residue_keys_with_ca) * esm_dim,
            dtype=torch.float32,
        ).reshape(len(residue_keys_with_ca), esm_dim) + 1.0
        outpath = self.embeddings_dir / f"{self.structure_path.stem}_chain_A_esmc.pt"
        if canonical:
            payload = build_embedding_payload(
                embedding,
                residue_keys_with_ca,
                structure_id=self.structure_path.stem,
                chain_id="A",
                source_path=str(self.structure_path),
            )
            torch.save(payload, outpath)
        else:
            torch.save(embedding, outpath)
        return outpath

    def test_load_esm_lookup_aligns_tensor_file_to_structure_residues(self) -> None:
        self.build_sample_embedding_file(esm_dim=8, canonical=False)
        structure = parse_structure_file(str(self.structure_path), structure_id=self.structure_path.stem)

        esm_lookup = load_esm_lookup_for_structure(
            structure=structure,
            structure_path=self.structure_path,
            embeddings_dir=self.embeddings_dir,
        )

        self.assertGreater(len(esm_lookup), 0)
        first_key = next(iter(esm_lookup))
        self.assertEqual(len(first_key), 3)
        self.assertEqual(tuple(esm_lookup[first_key].shape), (8,))
        self.assertGreater(float(esm_lookup[first_key].abs().sum().item()), 0.0)

    def test_load_esm_lookup_round_trips_canonical_payload(self) -> None:
        self.build_sample_embedding_file(esm_dim=8, canonical=True)
        structure = parse_structure_file(str(self.structure_path), structure_id=self.structure_path.stem)
        residue_keys, residue_keys_with_ca = residue_keys_for_structure_chain(structure, "A")

        esm_lookup = load_esm_lookup_for_structure(
            structure=structure,
            structure_path=self.structure_path,
            embeddings_dir=self.embeddings_dir,
        )

        self.assertEqual(sorted(esm_lookup), sorted(residue_keys_with_ca))
        self.assertNotEqual(len(residue_keys), 0)
        self.assertEqual(tuple(esm_lookup[residue_keys_with_ca[0]].shape), (8,))

    def test_training_loader_attaches_real_features(self) -> None:
        self.build_sample_embedding_file(esm_dim=8, canonical=True)

        result = load_training_pockets_with_report_from_dir(
            structure_dir=self.structure_root,
            require_full_labels=True,
            summary_csv=None,
            esm_dim=8,
            esm_embeddings_dir=self.embeddings_dir,
            require_esm_embeddings=True,
            external_features_root_dir=SAMPLE_JOB_DIR,
            require_external_features=True,
        )

        self.assertGreater(len(result.pockets), 0)
        first_pocket = result.pockets[0]
        self.assertTrue(all(residue.has_esm_embedding for residue in first_pocket.residues))
        self.assertTrue(all(residue.has_external_features for residue in first_pocket.residues))
        self.assertTrue(all(residue.esm_embedding is not None for residue in first_pocket.residues))
        self.assertTrue(
            all(float(residue.esm_embedding.abs().sum().item()) > 0.0 for residue in first_pocket.residues)
        )
        self.assertGreater(
            result.feature_report["esm_residue_coverage"],
            0.0,
        )
        self.assertGreater(
            result.feature_report["external_feature_residue_coverage"],
            0.0,
        )
        self.assertEqual(result.feature_report["feature_fallbacks"], [])

    def test_structure_dir_to_feature_lookup_parses_external_features(self) -> None:
        feature_lookup = structure_dir_to_feature_lookup(SAMPLE_FEATURE_DIR)

        self.assertGreater(len(feature_lookup), 0)
        sample_residue = next(iter(feature_lookup.values()))
        self.assertIn("SASA", sample_residue)
        self.assertIn("SASA_missing", sample_residue)
        self.assertIn("fa_sol", sample_residue)

    def test_training_loader_raises_when_required_esm_is_missing(self) -> None:
        with self.assertRaisesRegex(ValueError, "Missing required ESM embeddings"):
            load_training_pockets_with_report_from_dir(
                structure_dir=self.structure_root,
                require_full_labels=True,
                summary_csv=None,
                esm_dim=8,
                esm_embeddings_dir=self.embeddings_dir,
                require_esm_embeddings=True,
                external_features_root_dir=SAMPLE_JOB_DIR,
                require_external_features=True,
                invalid_structure_policy="error",
            )

class TrainingDataInvalidStructurePolicyTests(unittest.TestCase):
    @patch("training.data.load_structure_pockets")
    @patch("training.data.find_structure_files")
    def test_training_loader_skips_invalid_structures_when_configured(
        self,
        mock_find_structure_files,
        mock_load_structure_pockets,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            valid_structure = root / "1abc__chain_A__EC_1.1.1.1.pdb"
            invalid_structure = root / "2def__chain_A__EC_2.2.2.2.pdb"
            valid_structure.write_text("ATOM\n", encoding="utf-8")
            invalid_structure.write_text("ATOM\n", encoding="utf-8")
            mock_find_structure_files.return_value = [valid_structure, invalid_structure]

            valid_pocket = PocketRecord(
                structure_id=valid_structure.stem,
                pocket_id="valid-pocket",
                metal_element="ZN",
                metal_coords=[torch.tensor([0.0, 0.0, 0.0])],
                residues=[
                    ResidueRecord(
                        chain_id="A",
                        resseq=1,
                        icode="",
                        resname="HIS",
                        atoms={"CA": torch.tensor([0.0, 0.0, 0.0])},
                    )
                ],
                y_metal=0,
                y_ec=1,
            )
            mock_load_structure_pockets.side_effect = [
                ([valid_pocket], [], []),
                StructureLoadError("bad esm alignment"),
            ]

            result = load_training_pockets_with_report_from_dir(
                structure_dir=root,
                require_full_labels=True,
                summary_csv=None,
                esm_dim=8,
                esm_embeddings_dir=root / "embeddings",
                require_esm_embeddings=False,
                external_features_root_dir=root / "features",
                require_external_features=False,
                invalid_structure_policy="skip",
            )

            self.assertEqual(len(result.pockets), 1)
            self.assertEqual(result.feature_report["n_invalid_structures"], 1)
            self.assertEqual(
                result.feature_report["invalid_structures"][0]["structure_path"],
                str(invalid_structure),
            )


if __name__ == "__main__":
    unittest.main()
