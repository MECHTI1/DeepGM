from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from data_structures import PocketRecord, ResidueRecord
from training.site_filter import load_allowed_site_keys, pocket_matches_allowed_sites
from training.structure_loading import find_structure_files


def make_pocket_with_sites(*site_ids: tuple[str, int, str]) -> PocketRecord:
    return PocketRecord(
        structure_id="1abc__chain_A__EC_1.2.3.4",
        pocket_id="pocket-1",
        metal_element="ZN",
        metal_coords=[torch.tensor([0.0, 0.0, 0.0])],
        residues=[
            ResidueRecord(
                chain_id="A",
                resseq=10,
                icode="",
                resname="HIS",
                atoms={"CA": torch.tensor([1.0, 0.0, 0.0])},
            )
        ],
        metadata={"metal_site_ids": list(site_ids)},
    )


class TrainingSiteFilterTests(unittest.TestCase):
    def test_load_allowed_site_keys_normalizes_and_deduplicates_entries(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            summary_csv = Path(tmpdir) / "summary.csv"
            summary_csv.write_text(
                "\n".join(
                    [
                        "pdbid,metal residue number,EC number",
                        "1ABC,A_10,1.2.3.4;1.2.3.4",
                        "1abc,A_10,1.2.3.4",
                        "2DEF,B_4,2.7.11.1",
                    ]
                ),
                encoding="utf-8",
            )

            allowed_sites = load_allowed_site_keys(summary_csv)

            self.assertEqual(
                allowed_sites,
                {
                    ("1abc", "1.2.3.4", "A_10"),
                    ("2def", "2.7.11.1", "B_4"),
                },
            )

    def test_pocket_matches_allowed_sites_uses_structure_identity_and_site_ids(self) -> None:
        pocket = make_pocket_with_sites(("A", 10, ""), ("B", 15, ""))

        self.assertTrue(
            pocket_matches_allowed_sites(
                pocket,
                Path("1abc__chain_A__EC_1.2.3.4.pdb"),
                {("1abc", "1.2.3.4", "A_10")},
            )
        )
        self.assertFalse(
            pocket_matches_allowed_sites(
                pocket,
                Path("1abc__chain_A__EC_1.2.3.4.pdb"),
                {("1abc", "1.2.3.4", "C_99")},
            )
        )


class TrainingStructureLoadingTests(unittest.TestCase):
    def test_find_structure_files_skips_auxiliary_structure_directories(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            direct_structure = root / "keep_me.pdb"
            direct_structure.write_text("ATOM\n", encoding="utf-8")

            nested_root = root / "job_1"
            nested_root.mkdir()
            nested_structure = nested_root / "nested_keep.cif"
            nested_structure.write_text("data_test\n", encoding="utf-8")

            auxiliary_dir = root / "aux_structure"
            auxiliary_dir.mkdir()
            auxiliary_structure = auxiliary_dir / "aux_structure.pdb"
            auxiliary_structure.write_text("ATOM\n", encoding="utf-8")

            helper_dir = nested_root / "feature_outputs"
            helper_dir.mkdir()
            helper_structure = helper_dir / "ligand_fragment.pdb"
            helper_structure.write_text("ATOM\n", encoding="utf-8")

            structure_files = find_structure_files(root)

            self.assertEqual(structure_files, [nested_structure, direct_structure])


if __name__ == "__main__":
    unittest.main()
