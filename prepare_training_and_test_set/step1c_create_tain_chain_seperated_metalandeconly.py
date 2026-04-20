#!/usr/bin/env python3
from __future__ import annotations

import os
from pathlib import Path
import re
import sys

try:
    import gemmi
except ImportError:
    print("ERROR: gemmi is not installed.")
    print("Install it with:")
    print("  pip install gemmi")
    sys.exit(1)


# =========================================================
# INPUT / OUTPUT FOLDERS
# =========================================================
DATASET_ROOT = Path(
    os.getenv("DEEPGM_PINMYMETAL_SET_ROOT", "/media/Data/pinmymetal_sets/train")
).expanduser()
INPUT_OUTPUT_JOBS = [
    {
        "input_dir": Path(
            os.getenv("DEEPGM_STEP1C_INPUT_CIF_DIR", str(DATASET_ROOT / "cif_updated"))
        ).expanduser(),
        "output_dir": Path(
            os.getenv("DEEPGM_STEP1C_OUTPUT_CIF_DIR", str(DATASET_ROOT / "cif_updatedv2"))
        ).expanduser(),
        "input_kind": "cif",
        "output_format": "cif",
    },
    {
        "input_dir": Path(
            os.getenv("DEEPGM_STEP1C_INPUT_PDB_DIR", str(DATASET_ROOT / "pdb_updated"))
        ).expanduser(),
        "output_dir": Path(
            os.getenv("DEEPGM_STEP1C_OUTPUT_PDB_DIR", str(DATASET_ROOT / "pdb_updatedv2"))
        ).expanduser(),
        "input_kind": "pdb",
        "output_format": "pdb",
    },
]

OVERWRITE = True
# =========================================================


# Transition metals only
METAL_ELEMENTS = {
    "MN", "FE", "CO", "NI", "CU", "ZN"
}

WATER_NAMES = {"HOH", "WAT", "H2O", "DOD", "SOL"}
PROTEIN_RESIDUE_NAMES = {
    "ALA", "ARG", "ASN", "ASP", "CYS",
    "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO",
    "SER", "THR", "TRP", "TYR", "VAL",
    "ASX", "GLX", "SEC", "PYL", "MSE",
}


def is_water_residue(residue: gemmi.Residue) -> bool:
    return residue.name.strip().upper() in WATER_NAMES


def is_polymer_residue(residue: gemmi.Residue) -> bool:
    if residue.name.strip().upper() in PROTEIN_RESIDUE_NAMES:
        return True
    return residue.entity_type == gemmi.EntityType.Polymer


def residue_has_transition_metal(residue: gemmi.Residue) -> bool:
    for atom in residue:
        elem = atom.element.name.upper()
        if elem in METAL_ELEMENTS:
            return True
    return False


def is_supported_metal_ion_residue(residue: gemmi.Residue) -> bool:
    atoms = list(residue)
    if not atoms:
        return False
    return all(atom.element.name.upper() in METAL_ELEMENTS for atom in atoms)


def should_keep_residue(residue: gemmi.Residue) -> bool:
    if is_water_residue(residue):
        return False
    if is_polymer_residue(residue):
        return True
    return is_supported_metal_ion_residue(residue)


def chain_contains_transition_metal(chain: gemmi.Chain) -> bool:
    for residue in chain:
        if is_water_residue(residue):
            continue
        if residue_has_transition_metal(residue):
            return True
    return False


def read_structure(path: Path) -> gemmi.Structure:
    st = gemmi.read_structure(str(path))
    st.setup_entities()
    return st


# --------------------------------------------------
# PDB EC parsing from COMPND records
# --------------------------------------------------
def parse_pdb_chain_to_ec(path: Path) -> dict[str, str]:
    """
    Parse EC from PDB COMPND records like:

    COMPND   1 MOL_ID: 1;
    COMPND   2 MOLECULE: ...;
    COMPND   3 CHAIN: A, B;
    COMPND   4 EC: 1.2.3.4;
    """
    chain_to_ec: dict[str, str] = {}

    mol_blocks: dict[str, dict[str, str]] = {}
    current_mol_id = None

    mol_id_re = re.compile(r"MOL_ID:\s*([^;]+)\s*;", re.IGNORECASE)
    chain_re = re.compile(r"CHAIN:\s*([^;]+)\s*;", re.IGNORECASE)
    ec_re = re.compile(r"EC:\s*([^;]+)\s*;", re.IGNORECASE)

    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for raw_line in f:
                line = raw_line.rstrip("\n")

                if line.startswith(("ATOM  ", "HETATM", "MODEL ", "TER   ")):
                    break

                if not line.startswith("COMPND"):
                    continue

                payload = line[10:].strip()

                m = mol_id_re.search(payload)
                if m:
                    current_mol_id = m.group(1).strip()
                    mol_blocks.setdefault(current_mol_id, {})
                    continue

                if current_mol_id is None:
                    continue

                m = chain_re.search(payload)
                if m:
                    mol_blocks[current_mol_id]["CHAIN"] = m.group(1).strip()
                    continue

                m = ec_re.search(payload)
                if m:
                    mol_blocks[current_mol_id]["EC"] = m.group(1).strip()
                    continue

    except Exception as e:
        print(f"[WARN] Could not parse PDB COMPND EC info from {path.name}: {e}")
        return {}

    for _, fields in mol_blocks.items():
        chain_field = fields.get("CHAIN", "").strip()
        ec_field = fields.get("EC", "").strip()

        if not chain_field:
            continue
        if not ec_field:
            continue
        if ec_field.upper() == "NOT_AVAILABLE":
            continue
        if ec_field in {".", "?", "NONE", "NULL"}:
            continue

        for ch in chain_field.split(","):
            ch = ch.strip()
            if ch:
                chain_to_ec[ch] = ec_field

    return chain_to_ec


# --------------------------------------------------
# CIF EC parsing from custom header comments
# --------------------------------------------------
HEADER_CHAIN_EC_RE = re.compile(
    r"^#\s*MOL_ID:\s*[^;]+;\s*MOLECULE:\s*.*?;\s*CHAIN:\s*([^;]+?)\s*;\s*EC:\s*([^;]+?)\s*;",
    re.IGNORECASE,
)


def parse_cif_comment_chain_to_ec(path: Path) -> dict[str, str]:
    """
    Parse custom header comment lines like:
    # MOL_ID: 2; MOLECULE: ...; CHAIN: G; EC: 1.2.3.4; UNIPROT: ...
    """
    chain_to_ec: dict[str, str] = {}

    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for raw_line in f:
                line = raw_line.rstrip("\n")

                if (
                    line.startswith("data_")
                    or line.startswith("loop_")
                    or line.startswith("ATOM")
                    or line.startswith("HETATM")
                ):
                    break

                m = HEADER_CHAIN_EC_RE.match(line)
                if not m:
                    continue

                chain_field = m.group(1).strip()
                ec_field = m.group(2).strip()

                if not ec_field:
                    continue
                if ec_field.upper() == "NOT_AVAILABLE":
                    continue
                if ec_field in {".", "?", "NONE", "NULL"}:
                    continue

                for ch in chain_field.split(","):
                    ch = ch.strip()
                    if ch:
                        chain_to_ec[ch] = ec_field

    except Exception as e:
        print(f"[WARN] Could not parse CIF header EC info from {path.name}: {e}")

    return chain_to_ec


def parse_chain_to_ec(path: Path, input_kind: str) -> dict[str, str]:
    if input_kind == "pdb":
        return parse_pdb_chain_to_ec(path)
    elif input_kind == "cif":
        return parse_cif_comment_chain_to_ec(path)
    else:
        return {}


def sanitize_filename_fragment(value: str) -> str:
    sanitized = value.strip()
    sanitized = re.sub(r"\s+", "", sanitized)
    sanitized = sanitized.replace("/", "-")
    sanitized = re.sub(r"_+", "_", sanitized)
    sanitized = sanitized.strip("_")
    return sanitized or "unknown"


def build_structure_output_name(
    *,
    source_stem: str,
    chain_name: str,
    ec: str,
    suffix: str,
) -> str:
    safe_source_stem = sanitize_filename_fragment(source_stem)
    safe_chain_name = sanitize_filename_fragment(chain_name)
    # Preserve EC punctuation that downstream parsing already handles, but
    # remove whitespace and normalize separators inside the dynamic fragment so
    # the final name keeps exactly the intentional double-underscore delimiters.
    safe_ec = sanitize_filename_fragment(ec)
    return f"{safe_source_stem}__chain_{safe_chain_name}__EC_{safe_ec}{suffix}"


def make_single_chain_structure(
    original: gemmi.Structure,
    model_index: int,
    chain_name: str,
) -> gemmi.Structure:
    new_st = gemmi.Structure()
    new_st.name = original.name
    new_st.spacegroup_hm = original.spacegroup_hm
    new_st.cell = original.cell

    old_model = original[model_index]
    # gemmi API varies by version: some expose model.name, others only model.num
    model_id = getattr(old_model, "name", None)
    if not model_id:
        model_id = getattr(old_model, "num", model_index + 1)
    new_model = gemmi.Model(model_id)

    selected_chain = None
    for ch in old_model:
        if ch.name == chain_name:
            selected_chain = ch
            break

    if selected_chain is None:
        raise ValueError(f"Chain {chain_name!r} not found")

    new_chain = gemmi.Chain(selected_chain.name)

    for residue in selected_chain:
        # Keep the protein backbone and supported transition-metal ions, but
        # drop fragile small hetero residues that frequently crash MAHOMES /
        # Rosetta during atom completion.
        if not should_keep_residue(residue):
            continue

        new_res = gemmi.Residue()
        new_res.name = residue.name
        new_res.seqid = residue.seqid
        new_res.subchain = residue.subchain
        new_res.label_seq = residue.label_seq
        new_res.entity_type = residue.entity_type
        new_res.het_flag = residue.het_flag

        for atom in residue:
            new_atom = gemmi.Atom()
            new_atom.name = atom.name
            new_atom.element = atom.element
            new_atom.pos = atom.pos
            new_atom.occ = atom.occ
            new_atom.b_iso = atom.b_iso
            new_atom.altloc = atom.altloc
            new_atom.charge = atom.charge
            new_atom.serial = atom.serial
            new_res.add_atom(new_atom)

        new_chain.add_residue(new_res)

    new_model.add_chain(new_chain)
    new_st.add_model(new_model)
    new_st.setup_entities()
    return new_st


def add_ec_comment_to_cif(path: Path, ec: str) -> None:
    try:
        text = path.read_text(encoding="utf-8")
        lines = text.splitlines()
        insert_idx = 1 if lines else 0
        lines.insert(insert_idx, f"# EC_NUMBER: {ec}")
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    except Exception as e:
        print(f"[WARN] Could not add EC comment to {path.name}: {e}")


def add_ec_remark_to_pdb(path: Path, ec: str) -> None:
    try:
        text = path.read_text(encoding="utf-8")
        lines = text.splitlines()
        insert_idx = 1 if lines else 0
        lines.insert(insert_idx, f"REMARK EC_NUMBER: {ec}")
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    except Exception as e:
        print(f"[WARN] Could not add EC remark to {path.name}: {e}")


def write_structure(st: gemmi.Structure, out_path: Path, fmt: str, ec: str | None = None) -> None:
    if fmt == "cif":
        doc = st.make_mmcif_document()
        doc.write_file(str(out_path))
        if ec:
            add_ec_comment_to_cif(out_path, ec)
    elif fmt == "pdb":
        st.write_pdb(str(out_path))
        if ec:
            add_ec_remark_to_pdb(out_path, ec)
    else:
        raise ValueError(f"Unsupported output format: {fmt}")


def process_one_file(
    path: Path,
    output_dir: Path,
    input_kind: str,
    output_format: str,
    overwrite: bool,
) -> int:
    chain_to_ec = parse_chain_to_ec(path, input_kind=input_kind)

    if not chain_to_ec:
        print(f"[SKIP] {path.name}: no chain-level EC info found")
        return 0

    try:
        st = read_structure(path)
    except Exception as e:
        print(f"[ERROR] Failed reading {path.name}: {e}")
        return 0

    written = 0

    for model_index, model in enumerate(st):
        for chain in model:
            chain_name = chain.name

            if chain_name not in chain_to_ec:
                continue

            if not chain_contains_transition_metal(chain):
                continue

            ec = chain_to_ec[chain_name]

            try:
                out_st = make_single_chain_structure(st, model_index, chain_name)
            except Exception as e:
                print(f"[ERROR] Failed extracting chain {chain_name} from {path.name}: {e}")
                continue

            suffix = ".pdb" if output_format == "pdb" else ".cif"
            out_name = build_structure_output_name(
                source_stem=path.stem,
                chain_name=chain_name,
                ec=ec,
                suffix=suffix,
            )
            out_path = output_dir / out_name

            if out_path.exists() and not overwrite:
                print(f"[SKIP] Exists: {out_path.name}")
                continue

            try:
                write_structure(out_st, out_path, output_format, ec=ec)
                print(f"[OK] Wrote {out_path.name}")
                written += 1
            except Exception as e:
                print(f"[ERROR] Failed writing {out_path.name}: {e}")

    if written == 0:
        print(f"[SKIP] {path.name}: no chain satisfied both 'has transition metal' and 'has EC'")

    return written


def process_job(job: dict) -> int:
    input_dir = job["input_dir"]
    output_dir = job["output_dir"]
    input_kind = job["input_kind"]
    output_format = job["output_format"]

    output_dir.mkdir(parents=True, exist_ok=True)

    if input_kind == "cif":
        patterns = ("*.cif", "*.mmcif")
    elif input_kind == "pdb":
        patterns = ("*.pdb",)
    else:
        print(f"[ERROR] Unsupported input kind: {input_kind}")
        return 0

    input_files = []
    for pattern in patterns:
        input_files.extend(sorted(input_dir.glob(pattern)))

    if not input_files:
        print(f"[WARN] No files found in: {input_dir}")
        return 0

    print(f"\nProcessing {input_kind.upper()} files")
    print(f"Input : {input_dir}")
    print(f"Output: {output_dir}")

    total_written = 0
    for path in input_files:
        total_written += process_one_file(
            path=path,
            output_dir=output_dir,
            input_kind=input_kind,
            output_format=output_format,
            overwrite=OVERWRITE,
        )

    print(f"Finished {input_kind.upper()} job. Files written: {total_written}")
    return total_written


def main() -> None:
    grand_total = 0

    for job in INPUT_OUTPUT_JOBS:
        grand_total += process_job(job)

    print("\n========================================")
    print(f"All done. Total files written: {grand_total}")
    print("========================================")


if __name__ == "__main__":
    main()
