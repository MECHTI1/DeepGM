from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from build_colab_bundle import (
    StructureCheckResult,
    build_manifest_payload,
    collect_members_from_manifest,
    choose_tar_flags,
    load_manifest_payload,
    select_structure_members,
)


class TarFlagTests(unittest.TestCase):
    def test_choose_tar_flags_supports_expected_suffixes(self) -> None:
        self.assertEqual(choose_tar_flags(Path("bundle.tar.zst")), ["--zstd", "-cf"])
        self.assertEqual(choose_tar_flags(Path("bundle.tar.gz")), ["-czf"])
        self.assertEqual(choose_tar_flags(Path("bundle.tar")), ["-cf"])


class StructureMemberSelectionTests(unittest.TestCase):
    def test_select_structure_members_includes_structure_feature_dir_and_summary_parent(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            structure_root = Path(tmpdir) / "train_set"
            structure_path = structure_root / "job_0" / "1abc__chain_A__EC_1.1.1.1.pdb"
            feature_dir = structure_root / "job_0" / "1abc__chain_A__EC_1.1.1.1"
            summary_csv = structure_root / "data_summarizing_tables" / "summary.csv"

            feature_dir.mkdir(parents=True)
            structure_path.parent.mkdir(parents=True, exist_ok=True)
            structure_path.write_text("MODEL\n", encoding="utf-8")
            summary_csv.parent.mkdir(parents=True, exist_ok=True)
            summary_csv.write_text("pdbid,metal residue number,EC number\n", encoding="utf-8")

            members = select_structure_members(
                structure_path=structure_path,
                structure_root=structure_root,
                feature_root_dir=structure_root,
                summary_csv=summary_csv,
            )

            self.assertIn(structure_path, members)
            self.assertIn(feature_dir, members)
            self.assertIn(summary_csv.parent, members)


class ManifestPayloadTests(unittest.TestCase):
    def test_build_manifest_payload_summarizes_result_groups(self) -> None:
        results = [
            StructureCheckResult(
                structure_id="good",
                relative_structure_path="job_0/good.pdb",
                status="included",
                kept_pockets=2,
                feature_fallbacks=[],
                skipped_pockets=[],
            ),
            StructureCheckResult(
                structure_id="unused",
                relative_structure_path="job_0/unused.pdb",
                status="unused",
                kept_pockets=0,
                feature_fallbacks=[],
                skipped_pockets=[],
            ),
            StructureCheckResult(
                structure_id="bad",
                relative_structure_path="job_0/bad.pdb",
                status="invalid",
                kept_pockets=0,
                feature_fallbacks=[],
                skipped_pockets=[],
                error="boom",
            ),
        ]

        payload = build_manifest_payload(
            structure_dir=Path("/tmp/train_set"),
            summary_csv=Path("/tmp/train_set/data_summarizing_tables/summary.csv"),
            embeddings_dir=Path("/tmp/embeddings"),
            feature_root_dir=Path("/tmp/updated_feature_extraction"),
            results=results,
            excluded_structure_ids=["skip_me"],
            structure_archive_name="train_set_clean.tar.zst",
            embeddings_archive_name="embeddings_clean.tar.zst",
        )

        self.assertEqual(payload["n_total_structures"], 4)
        self.assertEqual(payload["n_included_structures"], 1)
        self.assertEqual(payload["n_unused_structures"], 1)
        self.assertEqual(payload["n_invalid_structures"], 1)
        self.assertEqual(payload["external_features_root_dir"], "/tmp/updated_feature_extraction")

    def test_load_manifest_payload_round_trips_json(self) -> None:
        payload = {"hello": "world", "count": 3}
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = Path(tmpdir) / "manifest.json"
            manifest_path.write_text('{"hello": "world", "count": 3}', encoding="utf-8")

            loaded = load_manifest_payload(manifest_path)

        self.assertEqual(loaded, payload)


class ManifestMemberCollectionTests(unittest.TestCase):
    def test_collect_members_from_manifest_uses_included_structures(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            structure_dir = root / "train_set"
            embeddings_dir = root / "embeddings"
            structure_path = structure_dir / "job_0" / "1abc__chain_A__EC_1.1.1.1.pdb"
            feature_dir = structure_dir / "job_0" / "1abc__chain_A__EC_1.1.1.1"
            summary_csv = structure_dir / "data_summarizing_tables" / "summary.csv"
            embedding_path = embeddings_dir / "1abc__chain_A__EC_1.1.1.1_chain_A_esmc.pt"

            feature_dir.mkdir(parents=True)
            structure_path.parent.mkdir(parents=True, exist_ok=True)
            structure_path.write_text("MODEL\n", encoding="utf-8")
            summary_csv.parent.mkdir(parents=True, exist_ok=True)
            summary_csv.write_text("pdbid,metal residue number,EC number\n", encoding="utf-8")
            embeddings_dir.mkdir(parents=True, exist_ok=True)
            embedding_path.write_text("placeholder", encoding="utf-8")

            manifest_payload = {
                "included_structures": [
                    {
                        "structure_id": "1abc__chain_A__EC_1.1.1.1",
                        "relative_structure_path": "job_0/1abc__chain_A__EC_1.1.1.1.pdb",
                    }
                ]
            }

            structure_members, embedding_members = collect_members_from_manifest(
                manifest_payload=manifest_payload,
                structure_dir=structure_dir,
                summary_csv=summary_csv,
                embeddings_dir=embeddings_dir,
                feature_root_dir=structure_dir,
            )

            self.assertIn(structure_path, structure_members)
            self.assertIn(feature_dir, structure_members)
            self.assertIn(summary_csv.parent, structure_members)
            self.assertIn(embedding_path, embedding_members)


if __name__ == "__main__":
    unittest.main()
