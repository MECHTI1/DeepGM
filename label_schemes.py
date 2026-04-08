from __future__ import annotations

EC_TOP_LEVEL_LABELS = {
    0: "Oxidoreductase",
    1: "Transferase",
    2: "Hydrolase",
    3: "Lyase",
    4: "Isomerase",
    5: "Ligase",
    6: "Translocase",
}
N_EC_CLASSES = len(EC_TOP_LEVEL_LABELS)

# Current metal training head:
# 0 -> Zn
# 1 -> Cu
# 2 -> Mn
# 3 -> merged Co / Fe / Ni class
METAL_TARGET_LABELS = {
    0: "Zn",
    1: "Cu",
    2: "Mn",
    3: "Co/Fe/Ni",
}
METAL_TARGET_MEMBERS = {
    0: ("Zn",),
    1: ("Cu",),
    2: ("Mn",),
    3: ("Co", "Fe", "Ni"),
}
N_METAL_CLASSES = len(METAL_TARGET_LABELS)

METAL_SYMBOL_TO_TARGET = {
    "ZN": 0,
    "CU": 1,
    "MN": 2,
    "CO": 3,
    "FE": 3,
    "NI": 3,
}

# Raw codes from prepare_training_and_test_set/pinmymetal_files/classmodel_train_set
# mapped into the 4-class metal target used for training.
CLASSMODEL_METAL_CODE_TO_TARGET = {
    7: 0,  # Zn
    6: 1,  # Cu
    1: 2,  # Mn
    2: 3,  # grouped Co / Fe / Ni class
}
CLASSMODEL_METAL_CODE_TO_LABEL = {
    raw_code: METAL_TARGET_LABELS[target_idx]
    for raw_code, target_idx in CLASSMODEL_METAL_CODE_TO_TARGET.items()
}


def map_classmodel_metal_code(raw_code: int) -> int:
    if raw_code not in CLASSMODEL_METAL_CODE_TO_TARGET:
        raise ValueError(
            f"Unsupported classmodel metal code {raw_code}. "
            f"Expected one of {sorted(CLASSMODEL_METAL_CODE_TO_TARGET)}."
        )
    return CLASSMODEL_METAL_CODE_TO_TARGET[raw_code]


def map_site_metal_symbol(symbol: str) -> int:
    normalized = symbol.strip().upper()
    if normalized not in METAL_SYMBOL_TO_TARGET:
        raise ValueError(
            f"Unsupported site metal symbol {symbol!r}. "
            f"Expected one of {sorted(METAL_SYMBOL_TO_TARGET)}."
        )
    return METAL_SYMBOL_TO_TARGET[normalized]


def map_site_metal_symbols(symbols: tuple[str, ...] | list[str]) -> int | None:
    target_ids = {map_site_metal_symbol(symbol) for symbol in symbols}
    if len(target_ids) != 1:
        return None
    return next(iter(target_ids))
