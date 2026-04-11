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

# Metal class mapping used by the current classifier:
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
N_METAL_CLASSES = len(METAL_TARGET_LABELS)

METAL_SYMBOL_TO_TARGET = {
    "ZN": 0,
    "CU": 1,
    "MN": 2,
    "CO": 3,
    "FE": 3,
    "NI": 3,
}
def map_site_metal_symbols(symbols: str | tuple[str, ...] | list[str]) -> int | None:
    if isinstance(symbols, str):
        symbols = (symbols,)

    target_ids = set()
    for symbol in symbols:
        try:
            target_ids.add(METAL_SYMBOL_TO_TARGET[symbol.strip().upper()])
        except KeyError as exc:
            raise ValueError(
                f"Unsupported site metal symbol {symbol!r}. "
                f"Expected one of {sorted(METAL_SYMBOL_TO_TARGET)}."
            ) from exc

    return next(iter(target_ids)) if len(target_ids) == 1 else None
