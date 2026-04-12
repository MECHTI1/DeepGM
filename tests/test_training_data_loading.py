from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

import torch

from graph.construction import parse_structure_file
from training.data import (
    load_esm_lookup_for_structure,
    load_training_pockets_with_report_from_dir,
    residue_keys_for_structure_chain,
)


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

    def build_sample_embedding_file(self, esm_dim: int = 8) -> Path:
        structure = parse_structure_file(str(self.structure_path), structure_id=self.structure_path.stem)
        residue_keys, residue_keys_with_ca = residue_keys_for_structure_chain(structure, "A")
        self.assertGreater(len(residue_keys), 0)
        self.assertGreater(len(residue_keys_with_ca), 0)

        embedding = torch.arange(
            len(residue_keys_with_ca) * esm_dim,
            dtype=torch.float32,
        ).reshape(len(residue_keys_with_ca), esm_dim) + 1.0
        outpath = self.embeddings_dir / f"{self.structure_path.stem}_chain_A_esmc.pt"
        torch.save(embedding, outpath)
        return outpath

    def test_load_esm_lookup_aligns_tensor_file_to_structure_residues(self) -> None:
        self.build_sample_embedding_file(esm_dim=8)
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

    def test_training_loader_attaches_real_features(self) -> None:
        self.build_sample_embedding_file(esm_dim=8)

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
            )


if __name__ == "__main__":
    unittest.main()
