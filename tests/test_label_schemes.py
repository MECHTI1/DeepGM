from __future__ import annotations

import csv
import unittest
from collections import Counter
from pathlib import Path

from label_schemes import METAL_TARGET_LABELS, N_METAL_CLASSES, map_site_metal_symbols


SUMMARY_CSV = Path(
    "/media/Data/pinmymetal_sets/mahomes/train_set/data_summarizing_tables/"
    "final_data_summarazing_table_transition_metals_only_catalytic.csv"
)


class LabelSchemeTests(unittest.TestCase):
    def test_active_metal_label_space_matches_current_dataset_intent(self) -> None:
        self.assertEqual(
            METAL_TARGET_LABELS,
            {
                0: "Mn",
                1: "Cu",
                2: "Zn",
                3: "Class VIII",
            },
        )
        self.assertEqual(N_METAL_CLASSES, 4)

    def test_supported_symbols_map_to_active_targets(self) -> None:
        self.assertEqual(map_site_metal_symbols("MN"), 0)
        self.assertEqual(map_site_metal_symbols("CU"), 1)
        self.assertEqual(map_site_metal_symbols("ZN"), 2)
        self.assertEqual(map_site_metal_symbols("CO"), 3)
        self.assertEqual(map_site_metal_symbols("FE"), 3)
        self.assertEqual(map_site_metal_symbols("NI"), 3)

    def test_unsupported_symbol_can_be_skipped_explicitly(self) -> None:
        self.assertIsNone(map_site_metal_symbols("MO", unsupported_policy="skip"))

    def test_current_summary_metals_stay_within_supported_runtime_set(self) -> None:
        if not SUMMARY_CSV.is_file():
            raise unittest.SkipTest(f"Missing summary CSV: {SUMMARY_CSV}")

        with SUMMARY_CSV.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))

        counts = Counter(row["metal residue type"].strip().upper() for row in rows)
        self.assertTrue(set(counts).issubset({"MN", "FE", "CO", "NI", "CU", "ZN"}))


if __name__ == "__main__":
    unittest.main()
