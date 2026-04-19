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

# Active metal class mapping:
# 0 -> Mn
# 1 -> Cu
# 2 -> Zn
# 3 -> Class VIII (grouped Fe / Co / Ni)
METAL_TARGET_LABELS = {
    0: "Mn",
    1: "Cu",
    2: "Zn",
    3: "Class VIII",
}
N_METAL_CLASSES = len(METAL_TARGET_LABELS)

METAL_SYMBOL_TO_TARGET = {
    "MN": 0,
    "CU": 1,
    "ZN": 2,
    "FE": 3,
    "CO": 3,
    "NI": 3,
}


def map_site_metal_symbols(
    symbols: str | tuple[str, ...] | list[str],
    *,
    unsupported_metal_policy: str = "error",
) -> int | None:
    if isinstance(symbols, str):
        symbols = (symbols,)

    target_ids = set()
    for symbol in symbols:
        normalized = symbol.strip().upper()
        try:
            target_ids.add(METAL_SYMBOL_TO_TARGET[normalized])
        except KeyError as exc:
            if unsupported_metal_policy == "skip":
                return None
            raise ValueError(
                f"Unsupported site metal symbol {symbol!r}. "
                f"Expected one of {sorted(METAL_SYMBOL_TO_TARGET)}."
            ) from exc

    return next(iter(target_ids)) if len(target_ids) == 1 else None
