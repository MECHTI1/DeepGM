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
                0: "Zn",
                1: "Cu",
                2: "Co/Fe/Ni",
            },
        )
        self.assertEqual(N_METAL_CLASSES, 3)

    def test_supported_symbols_map_to_active_targets(self) -> None:
        self.assertEqual(map_site_metal_symbols("ZN"), 0)
        self.assertEqual(map_site_metal_symbols("CU"), 1)
        self.assertEqual(map_site_metal_symbols("CO"), 2)
        self.assertEqual(map_site_metal_symbols("FE"), 2)
        self.assertEqual(map_site_metal_symbols("NI"), 2)

    def test_current_summary_has_no_mn_rows(self) -> None:
        if not SUMMARY_CSV.is_file():
            raise unittest.SkipTest(f"Missing summary CSV: {SUMMARY_CSV}")

        with SUMMARY_CSV.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))

        counts = Counter(row["metal residue type"].strip().upper() for row in rows)
        self.assertEqual(counts.get("MN", 0), 0)
        self.assertEqual(sorted(counts), ["CO", "CU", "FE", "NI", "ZN"])


if __name__ == "__main__":
    unittest.main()
