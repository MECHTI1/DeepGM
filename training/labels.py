from __future__ import annotations

from pathlib import Path
import re
from typing import Optional, Tuple

from data_structures import PocketRecord
from label_schemes import map_site_metal_symbols


EC_TOP_LEVEL_RE = re.compile(r"__EC_(\d+)")
STRUCTURE_ID_RE = re.compile(r"^(?P<pdbid>[^_]+)__chain_(?P<chain>[^_]+)__EC_(?P<ec>.+)$")


def normalize_ec_numbers(value: str) -> str:
    values = []
    seen = set()
    for ec in re.split(r"[;,]", value):
        ec = ec.strip()
        if not ec or ec in seen:
            continue
        seen.add(ec)
        values.append(ec)
    return ";".join(values)


def parse_structure_identity(structure_id: str) -> Tuple[str, str, str]:
    match = STRUCTURE_ID_RE.match(structure_id.strip())
    if match is None:
        raise ValueError(f"Could not parse structure identity from {structure_id!r}")
    return (
        match.group("pdbid").strip().lower(),
        match.group("chain").strip(),
        normalize_ec_numbers(match.group("ec")),
    )


def parse_ec_top_level_from_structure_path(path: Path) -> Optional[int]:
    match = EC_TOP_LEVEL_RE.search(path.stem)
    if not match:
        return None
    top_level = int(match.group(1))
    if not 1 <= top_level <= 7:
        return None
    return top_level - 1


def infer_metal_target_class_from_pocket(pocket: PocketRecord) -> Optional[int]:
    observed_symbols = pocket.metadata.get("metal_symbols_observed")
    if isinstance(observed_symbols, list) and observed_symbols:
        return map_site_metal_symbols(observed_symbols)

    metal_element = pocket.metal_element.strip()
    if not metal_element:
        return None

    return map_site_metal_symbols(metal_element)
