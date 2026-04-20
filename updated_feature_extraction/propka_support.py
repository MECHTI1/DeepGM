from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

from biotite.structure.io import load_structure, save_structure


@dataclass(frozen=True)
class PropkaResidueFeatures:
    predicted_pka: float
    model_pka: float
    buried_percent: float
    dpka_desolv: float
    dpka_bg: float
    dpka_titr: float


@dataclass(frozen=True)
class PropkaRunResult:
    residues: Dict[tuple[str, int, str], PropkaResidueFeatures]
    warnings: list[str]


def _primary_row_tokens(tokens: list[str]) -> bool:
    return len(tokens) >= 19 and tokens[5] == "%"


def _continuation_row_tokens(tokens: list[str]) -> bool:
    return len(tokens) >= 12 and "%" not in tokens[:8]


def _summary_row_tokens(tokens: list[str]) -> bool:
    if len(tokens) < 5 or tokens[0] in {"Group", "SUMMARY", "Free", "The", "Could", "or"}:
        return False
    try:
        int(tokens[1])
    except ValueError:
        return False
    return True


def _looks_like_residue_key(tokens: list[str]) -> bool:
    if len(tokens) < 3:
        return False
    try:
        int(tokens[1])
    except ValueError:
        return False
    return True


def parse_propka_output_text(text: str) -> Dict[tuple[str, int, str], PropkaResidueFeatures]:
    detail_totals: dict[tuple[str, int, str], dict[str, float]] = {}
    summary_model_pka: dict[tuple[str, int, str], float] = {}
    in_detail_table = False
    in_summary_table = False

    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()
        if not stripped:
            continue

        if stripped.startswith("RESIDUE"):
            in_detail_table = True
            in_summary_table = False
            continue
        if stripped.startswith("SUMMARY OF THIS PREDICTION"):
            in_detail_table = False
            in_summary_table = True
            continue
        if stripped.startswith("Group      pKa"):
            continue
        if stripped.startswith("---------") or stripped.startswith("-----------") or stripped.startswith("---"):
            continue
        if stripped.startswith("Coupled residues") or stripped.startswith("Free energy of"):
            in_summary_table = False
            continue

        tokens = stripped.split()
        if in_detail_table:
            if not _looks_like_residue_key(tokens):
                continue
            resname = tokens[0]
            if resname in {"N+", "C-"}:
                continue
            key = (tokens[2], int(tokens[1]), resname)
            entry = detail_totals.setdefault(
                key,
                {
                    "predicted_pka": 0.0,
                    "buried_percent": 0.0,
                    "dpka_desolv": 0.0,
                    "dpka_bg": 0.0,
                    "dpka_titr": 0.0,
                    "saw_primary": 0.0,
                },
            )

            if _primary_row_tokens(tokens):
                predicted_pka = float(tokens[3].rstrip("*"))
                buried_percent = float(tokens[4])
                regular = float(tokens[6])
                regular_re = float(tokens[8])
                sidechain_hbond = float(tokens[10])
                backbone_hbond = float(tokens[14])
                coulombic = float(tokens[18])
                entry["predicted_pka"] = predicted_pka
                entry["buried_percent"] = buried_percent
                entry["dpka_desolv"] += regular + regular_re
                entry["dpka_bg"] += sidechain_hbond + backbone_hbond
                entry["dpka_titr"] += coulombic
                entry["saw_primary"] = 1.0
                continue

            if _continuation_row_tokens(tokens):
                sidechain_hbond = float(tokens[3])
                backbone_hbond = float(tokens[7])
                coulombic = float(tokens[11])
                entry["dpka_bg"] += sidechain_hbond + backbone_hbond
                entry["dpka_titr"] += coulombic
            continue

        if in_summary_table and _summary_row_tokens(tokens):
            resname = tokens[0]
            if resname in {"N+", "C-"} or len(tokens) < 5:
                continue
            key = (tokens[2], int(tokens[1]), resname)
            summary_model_pka[key] = float(tokens[4])

    parsed: Dict[tuple[str, int, str], PropkaResidueFeatures] = {}
    for key, values in detail_totals.items():
        if not values["saw_primary"]:
            continue
        model_pka = summary_model_pka.get(key, values["predicted_pka"])
        parsed[key] = PropkaResidueFeatures(
            predicted_pka=values["predicted_pka"],
            model_pka=model_pka,
            buried_percent=values["buried_percent"],
            dpka_desolv=values["dpka_desolv"],
            dpka_bg=values["dpka_bg"],
            dpka_titr=values["dpka_titr"],
        )
    return parsed


def _prepare_propka_input_path(structure_path: Path, temp_dir: Path) -> Path:
    if structure_path.suffix.lower() == ".pdb":
        target = temp_dir / structure_path.name
        shutil.copy2(structure_path, target)
        return target

    atom_array = load_structure(str(structure_path))
    target = temp_dir / f"{structure_path.stem}.pdb"
    save_structure(str(target), atom_array)
    return target


def run_propka_for_structure(structure_path: Path, *, ph: float = 7.0) -> PropkaRunResult:
    with tempfile.TemporaryDirectory(prefix="deepgm_propka_") as tmpdir:
        temp_dir = Path(tmpdir)
        propka_input = _prepare_propka_input_path(structure_path, temp_dir)
        command = [
            sys.executable,
            "-m",
            "propka",
            "-q",
            "-o",
            str(ph),
            str(propka_input.name),
        ]
        completed = subprocess.run(
            command,
            cwd=temp_dir,
            capture_output=True,
            text=True,
            check=False,
        )
        warnings = [
            line.strip()
            for line in (completed.stdout.splitlines() + completed.stderr.splitlines())
            if line.strip()
        ]
        if completed.returncode != 0:
            raise RuntimeError(
                f"PROPKA failed for {structure_path} with exit code {completed.returncode}: "
                f"{' | '.join(warnings[:8])}"
            )

        output_path = temp_dir / f"{propka_input.stem}.pka"
        if not output_path.is_file():
            raise FileNotFoundError(f"Expected PROPKA output was not created for {structure_path}")

        residues = parse_propka_output_text(output_path.read_text(encoding="utf-8", errors="ignore"))
        return PropkaRunResult(residues=residues, warnings=warnings)
