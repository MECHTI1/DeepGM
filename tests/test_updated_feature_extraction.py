from __future__ import annotations

import json
import tempfile
import unittest
from concurrent.futures import Future
from pathlib import Path
from unittest.mock import patch

from training.feature_sources import resolve_structure_feature_dir
from training.external_feature_loading import structure_dir_to_feature_lookup
from training.feature_paths import resolve_runtime_feature_paths
from updated_feature_extraction import generate_features
from updated_feature_extraction.propka_support import _prepare_propka_input_path, parse_propka_output_text


class UpdatedFeatureExtractionTests(unittest.TestCase):
    def test_structure_dir_to_feature_lookup_parses_updated_json_format(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            feature_dir = Path(tmpdir) / "1abc__chain_A__EC_1.1.1.1"
            feature_dir.mkdir(parents=True)
            payload = {
                "schema_version": 1,
                "residues": [
                    {
                        "chain_id": "A",
                        "resseq": 10,
                        "icode": "",
                        "features": {
                            "SASA": 12.5,
                            "fa_sol": 0.75,
                            "fa_rep": 0.25,
                        },
                    }
                ],
            }
            (feature_dir / "residue_features.json").write_text(json.dumps(payload), encoding="utf-8")

            feature_lookup = structure_dir_to_feature_lookup(feature_dir)

            self.assertEqual(sorted(feature_lookup), [("A", 10, "")])
            self.assertEqual(feature_lookup[("A", 10, "")]["SASA"], 12.5)
            self.assertEqual(feature_lookup[("A", 10, "")]["fa_sol"], 0.75)
            self.assertEqual(feature_lookup[("A", 10, "")]["fa_rep"], 0.25)
            self.assertEqual(feature_lookup[("A", 10, "")]["BSA"], 0.0)
            self.assertEqual(feature_lookup[("A", 10, "")]["SASA_missing"], 0.0)

    def test_parse_propka_output_text_aggregates_primary_and_continuation_rows(self) -> None:
        text = """
---------  -----   ------   ---------------------    --------------    --------------    --------------
                            DESOLVATION  EFFECTS       SIDECHAIN          BACKBONE        COULOMBIC
 RESIDUE    pKa    BURIED     REGULAR      RE        HYDROGEN BOND     HYDROGEN BOND      INTERACTION
---------  -----   ------   ---------   ---------    --------------    --------------    --------------
ASP  31 A   4.12     0 %    0.28  159   0.00    0    0.00 XXX   0 X    0.00 XXX   0 X   -0.06 ARG  28 A
ASP  31 A                                            0.00 XXX   0 X    0.00 XXX   0 X    0.10 ASP  38 A

SUMMARY OF THIS PREDICTION
       Group      pKa  model-pKa   ligand atom-type
   ASP  31 A     4.12       3.80
"""
        parsed = parse_propka_output_text(text)

        residue = parsed[("A", 31, "ASP")]
        self.assertAlmostEqual(residue.predicted_pka, 4.12)
        self.assertAlmostEqual(residue.model_pka, 3.80)
        self.assertAlmostEqual(residue.dpka_desolv, 0.28)
        self.assertAlmostEqual(residue.dpka_bg, 0.0)
        self.assertAlmostEqual(residue.dpka_titr, 0.04)

    def test_prepare_propka_input_path_strips_nonmetal_hetero_groups_but_keeps_metals(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "sample.pdb"
            source.write_text(
                "\n".join(
                    [
                        "ATOM      1  N   GLY A   1      11.000  12.000  13.000  1.00 20.00           N  ",
                        "HETATM    2 FE   FE  A 101      10.000  11.000  12.000  1.00 20.00          FE  ",
                        "HETATM    3  C1 LIG A 201      14.000  15.000  16.000  1.00 20.00           C  ",
                        "TER",
                        "END",
                        "",
                    ]
                ),
                encoding="utf-8",
            )

            prepared = _prepare_propka_input_path(source, root / "out")
            prepared_text = prepared.read_text(encoding="utf-8")

            self.assertIn("ATOM      1", prepared_text)
            self.assertIn("HETATM    2 FE   FE", prepared_text)
            self.assertNotIn("LIG", prepared_text)

    def test_resolve_runtime_feature_paths_prefers_updated_feature_dir_when_present(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            structure_dir = Path(tmpdir) / "structures"
            structure_dir.mkdir()
            updated_feature_dir = Path(tmpdir) / "updated_features"
            updated_feature_dir.mkdir()

            with patch(
                "training.feature_paths.get_default_updated_feature_extraction_dir",
                return_value=updated_feature_dir,
            ):
                embeddings_dir, feature_root_dir = resolve_runtime_feature_paths(
                    structure_dir=structure_dir,
                    esm_embeddings_dir=None,
                    external_features_root_dir=None,
                )

            self.assertEqual(feature_root_dir, updated_feature_dir)
            self.assertIsInstance(embeddings_dir, Path)

    def test_resolve_runtime_feature_paths_uses_structure_dir_for_bluues_rosetta_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            structure_dir = Path(tmpdir) / "structures"
            structure_dir.mkdir()

            embeddings_dir, feature_root_dir = resolve_runtime_feature_paths(
                structure_dir=structure_dir,
                esm_embeddings_dir=None,
                external_features_root_dir=None,
                external_feature_source="bluues_rosetta",
            )

            self.assertEqual(feature_root_dir, structure_dir)
            self.assertIsInstance(embeddings_dir, Path)

    def test_structure_dir_resolution_prefers_updated_root_over_legacy_dir_in_auto_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            structure_root = root / "structures"
            structure_root.mkdir()
            structure_path = structure_root / "1abc__chain_A__EC_1.1.1.1.pdb"
            structure_path.write_text("ATOM\n", encoding="utf-8")

            legacy_dir = structure_root / structure_path.stem
            legacy_dir.mkdir()
            (legacy_dir / f"{structure_path.stem}.pdb").write_text("ATOM\n", encoding="utf-8")

            updated_root = root / "updated"
            updated_dir = updated_root / structure_path.stem
            updated_dir.mkdir(parents=True)
            payload = {
                "schema_version": 1,
                "residues": [
                    {
                        "chain_id": "A",
                        "resseq": 10,
                        "icode": "",
                        "features": {"SASA": 5.0},
                    }
                ],
            }
            (updated_dir / "residue_features.json").write_text(json.dumps(payload), encoding="utf-8")

            resolved_dir = resolve_structure_feature_dir(
                structure_path=structure_path,
                structure_root=structure_root,
                feature_root_dir=updated_root,
                external_feature_source="auto",
            )

            self.assertEqual(resolved_dir, updated_dir)

    @patch("updated_feature_extraction.generate_features.build_structure_feature_payload")
    @patch("updated_feature_extraction.generate_features.find_structure_files")
    def test_generate_features_continues_after_per_structure_failure(
        self,
        mock_find_structure_files,
        mock_build_payload,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            structure_dir = Path(tmpdir) / "structures"
            output_dir = Path(tmpdir) / "output"
            structure_dir.mkdir()
            good = structure_dir / "good.pdb"
            bad = structure_dir / "bad.pdb"
            good.write_text("ATOM\n", encoding="utf-8")
            bad.write_text("ATOM\n", encoding="utf-8")
            mock_find_structure_files.return_value = [good, bad]

            def build_payload(path, **_kwargs):
                if Path(path).stem == "bad":
                    raise RuntimeError("boom")
                return {
                    "schema_version": 1,
                    "residues": [],
                }

            mock_build_payload.side_effect = build_payload

            generate_features.main(
                [
                    "--structure-dir",
                    str(structure_dir),
                    "--output-root",
                    str(output_dir),
                ]
            )

            self.assertTrue((output_dir / "good" / "residue_features.json").is_file())
            failures = json.loads((output_dir / "generation_failures.json").read_text(encoding="utf-8"))
            self.assertEqual(len(failures), 1)
            self.assertEqual(Path(failures[0]["structure_path"]).name, "bad.pdb")
            self.assertEqual(failures[0]["error"], "boom")

    @patch("updated_feature_extraction.generate_features.as_completed", side_effect=lambda futures: list(futures))
    @patch("updated_feature_extraction.generate_features.find_structure_files")
    @patch("updated_feature_extraction.generate_features.generate_feature_file_for_structure")
    @patch("updated_feature_extraction.generate_features.ProcessPoolExecutor")
    def test_generate_features_parallel_mode_collects_successes_and_failures(
        self,
        mock_executor_cls,
        mock_generate_feature_file_for_structure,
        mock_find_structure_files,
        _mock_as_completed,
    ) -> None:
        class FakeExecutor:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def submit(self, fn, *args, **kwargs):
                future = Future()
                try:
                    future.set_result(fn(*args, **kwargs))
                except Exception as exc:
                    future.set_exception(exc)
                return future

        with tempfile.TemporaryDirectory() as tmpdir:
            structure_dir = Path(tmpdir) / "structures"
            output_dir = Path(tmpdir) / "output"
            structure_dir.mkdir()
            good = structure_dir / "good.pdb"
            bad = structure_dir / "bad.pdb"
            good.write_text("ATOM\n", encoding="utf-8")
            bad.write_text("ATOM\n", encoding="utf-8")
            mock_find_structure_files.return_value = [good, bad]
            mock_executor_cls.return_value = FakeExecutor()

            def generate_feature_file_side_effect(path, **_kwargs):
                if Path(path).stem == "bad":
                    raise RuntimeError("boom")
                saved_path = output_dir / Path(path).stem / "residue_features.json"
                saved_path.parent.mkdir(parents=True, exist_ok=True)
                saved_path.write_text("{}", encoding="utf-8")
                return saved_path

            mock_generate_feature_file_for_structure.side_effect = generate_feature_file_side_effect

            generate_features.main(
                [
                    "--structure-dir",
                    str(structure_dir),
                    "--output-root",
                    str(output_dir),
                    "--jobs",
                    "2",
                ]
            )

            self.assertTrue((output_dir / "good" / "residue_features.json").is_file())
            failures = json.loads((output_dir / "generation_failures.json").read_text(encoding="utf-8"))
            self.assertEqual(len(failures), 1)
            self.assertEqual(Path(failures[0]["structure_path"]).name, "bad.pdb")
            self.assertEqual(failures[0]["error"], "boom")


if __name__ == "__main__":
    unittest.main()
