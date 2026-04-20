from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from data_structures import PocketRecord
from evaluate_legacy_test_set import (
    LEGACY_LABEL_TO_TARGET,
    aggregate_structure_logits,
    load_legacy_test_labels,
    resolve_legacy_structure_path,
)


def make_pocket(structure_id: str, label_id: int) -> PocketRecord:
    pocket = PocketRecord(
        structure_id=structure_id,
        pocket_id=f"{structure_id}_METAL_0",
        metal_element="ZN",
        metal_coords=[torch.tensor([0.0, 0.0, 0.0])],
        residues=[],
        y_metal=label_id,
    )
    return pocket


class LegacyLabelLoadingTests(unittest.TestCase):
    def test_load_legacy_test_labels_maps_supported_codes_and_skips_mixed(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "test.csv"
            csv_path.write_text(
                "\n".join(
                    [
                        "pdbid,label_metal",
                        "1abc,1",
                        "1abc,1",
                        "2def,6",
                        "3ghi,2",
                        "4jkl,7",
                        "5mno,1",
                        "5mno,7",
                        "6pqr,99",
                    ]
                ),
                encoding="utf-8",
            )

            result = load_legacy_test_labels(csv_path)

        self.assertEqual(
            result.label_by_pdbid,
            {
                "1abc": LEGACY_LABEL_TO_TARGET["1"],
                "2def": LEGACY_LABEL_TO_TARGET["6"],
                "3ghi": LEGACY_LABEL_TO_TARGET["2"],
                "4jkl": LEGACY_LABEL_TO_TARGET["7"],
            },
        )
        self.assertEqual(result.mixed_pdbids, {"5mno": ["1", "7"]})
        self.assertEqual(result.unsupported_label_counts, {"99": 1})


class LegacyStructureResolutionTests(unittest.TestCase):
    def test_resolve_legacy_structure_path_prefers_subdirectories(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            cif_dir = root / "cif"
            cif_dir.mkdir()
            target = cif_dir / "1abc.cif"
            target.write_text("data_1abc\n", encoding="utf-8")

            resolved = resolve_legacy_structure_path(root, "1abc")

        self.assertEqual(resolved, target)


class StructureAggregationTests(unittest.TestCase):
    def test_aggregate_structure_logits_averages_per_structure(self) -> None:
        pockets = [
            make_pocket("1abc", 2),
            make_pocket("1abc", 2),
            make_pocket("2def", 1),
        ]
        metal_logits = torch.tensor(
            [
                [0.0, 1.0, 3.0, 0.0],
                [0.0, 1.0, 1.0, 0.0],
                [0.0, 4.0, 1.0, 0.0],
            ],
            dtype=torch.float32,
        )
        ec_logits = torch.tensor(
            [
                [1.0, 3.0, 0.0],
                [1.0, 1.0, 0.0],
                [4.0, 1.0, 0.0],
            ],
            dtype=torch.float32,
        )

        structure_ids, structure_metal_logits, structure_ec_logits, structure_y, pocket_counts = (
            aggregate_structure_logits(
            pockets=pockets,
            metal_logits=metal_logits,
            ec_logits=ec_logits,
        ))

        self.assertEqual(structure_ids, ["1abc", "2def"])
        self.assertTrue(torch.equal(structure_y, torch.tensor([2, 1], dtype=torch.long)))
        self.assertEqual(pocket_counts, {"1abc": 2, "2def": 1})
        self.assertTrue(torch.allclose(structure_metal_logits[0], torch.tensor([0.0, 1.0, 2.0, 0.0])))
        self.assertTrue(torch.allclose(structure_metal_logits[1], torch.tensor([0.0, 4.0, 1.0, 0.0])))
        self.assertTrue(torch.allclose(structure_ec_logits[0], torch.tensor([1.0, 2.0, 0.0])))
        self.assertTrue(torch.allclose(structure_ec_logits[1], torch.tensor([4.0, 1.0, 0.0])))


if __name__ == "__main__":
    unittest.main()
