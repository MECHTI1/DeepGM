from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from data_structures import PocketRecord, ResidueRecord
from training.labels import normalize_ec_number_list
from training.site_filter import (
    load_allowed_site_metal_labels,
    matched_site_metal_types,
    pocket_matches_allowed_sites,
)
from training.structure_loading import find_structure_files, load_structure_pockets


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
    def test_normalize_ec_number_list_canonicalizes_separators_and_deduplicates(self) -> None:
        self.assertEqual(
            normalize_ec_number_list("1.2.3.4, 2.7.11.1;1.2.3.4"),
            "1.2.3.4;2.7.11.1",
        )

    def test_load_allowed_site_metal_labels_keeps_site_specific_metal_identity(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            summary_csv = Path(tmpdir) / "summary.csv"
            summary_csv.write_text(
                "\n".join(
                    [
                        "pdbid,metal residue number,metal residue type,EC number",
                        "1ABC,A_10,zn,1.2.3.4",
                        "2DEF,B_4,Cu,2.7.11.1",
                    ]
                ),
                encoding="utf-8",
            )

            metal_labels = load_allowed_site_metal_labels(summary_csv)

            self.assertEqual(
                metal_labels,
                {
                    ("1abc", "1.2.3.4", "A_10"): "ZN",
                    ("2def", "2.7.11.1", "B_4"): "CU",
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

    def test_matched_site_metal_types_uses_summary_site_labels(self) -> None:
        pocket = make_pocket_with_sites(("A", 10, ""), ("B", 15, ""))

        self.assertEqual(
            matched_site_metal_types(
                pocket,
                Path("1abc__chain_A__EC_1.2.3.4.pdb"),
                {
                    ("1abc", "1.2.3.4", "A_10"): "CU",
                    ("1abc", "1.2.3.4", "B_15"): "ZN",
                },
            ),
            {"CU", "ZN"},
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

    @patch("training.structure_loading.attach_structure_features_to_pocket")
    @patch("training.structure_loading.load_structure_feature_sources", return_value=object())
    @patch("training.structure_loading.extract_metal_pockets_from_structure")
    @patch("training.structure_loading.parse_structure_file", return_value=object())
    def test_load_structure_pockets_uses_summary_site_label_for_mixed_metal_pocket(
        self,
        _mock_parse_structure_file,
        mock_extract_metal_pockets,
        _mock_load_structure_feature_sources,
        _mock_attach_structure_features_to_pocket,
    ) -> None:
        pocket = make_pocket_with_sites(("A", 601, ""), ("A", 602, ""))
        pocket.metadata["metal_symbols_observed"] = ["CO", "ZN"]
        mock_extract_metal_pockets.return_value = [pocket]

        pockets, _feature_fallbacks, skipped_pockets = load_structure_pockets(
            structure_path=Path("1gyt__chain_A__EC_3.4.11.1.pdb"),
            structure_root=Path("."),
            allowed_site_metal_labels={("1gyt", "3.4.11.1", "A_601"): "CO"},
            esm_dim=8,
            embeddings_dir=Path("."),
            require_esm_embeddings=False,
            feature_root_dir=Path("."),
            require_external_features=False,
            unsupported_metal_policy="error",
        )

        self.assertEqual(skipped_pockets, [])
        self.assertEqual(len(pockets), 1)
        self.assertEqual(pockets[0].y_metal, 3)
        self.assertEqual(pockets[0].y_ec, 2)
        self.assertEqual(pockets[0].metadata["matched_summary_site_metal_types"], ["CO"])


if __name__ == "__main__":
    unittest.main()
