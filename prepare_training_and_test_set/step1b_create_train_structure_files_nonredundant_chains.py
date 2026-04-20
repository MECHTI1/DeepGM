#!/usr/bin/env python3
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional

from Bio.PDB import MMCIFParser, MMCIFIO, Select
from Bio.PDB.MMCIF2Dict import MMCIF2Dict


# =========================================================
# PATHS
# =========================================================
DATASET_ROOT = Path(
    os.getenv("DEEPGM_PINMYMETAL_SET_ROOT", "/media/Data/pinmymetal_sets/train")
).expanduser()
INPUT_DIR = Path(os.getenv("DEEPGM_STEP1B_INPUT_DIR", str(DATASET_ROOT / "cif"))).expanduser()
OUTPUT_CIF_DIR = Path(
    os.getenv("DEEPGM_STEP1B_OUTPUT_CIF_DIR", str(DATASET_ROOT / "cif_updated"))
).expanduser()
OUTPUT_PDB_DIR = Path(
    os.getenv("DEEPGM_STEP1B_OUTPUT_PDB_DIR", str(DATASET_ROOT / "pdb_updated"))
).expanduser()

OVERWRITE = False
VERBOSE = True
# =========================================================


def log(msg: str) -> None:
    if VERBOSE:
        print(msg)


def find_cif_files(input_dir: Path) -> List[Path]:
    files = sorted(input_dir.glob("*.cif")) + sorted(input_dir.glob("*.mmcif"))
    seen = set()
    unique = []
    for f in files:
        if f not in seen:
            unique.append(f)
            seen.add(f)
    return unique


def mmcif_get_list(mmcif_dict: dict, key: str) -> Optional[List[str]]:
    value = mmcif_dict.get(key)
    if value is None:
        return None
    if isinstance(value, str):
        return [value]
    return list(value)


def load_structure(cif_path: Path):
    """
    Parse mmCIF using author chain IDs and author residue numbering.
    """
    parser = MMCIFParser(
        QUIET=True,
        auth_chains=True,
        auth_residues=True,
    )
    structure_id = cif_path.stem
    structure = parser.get_structure(structure_id, str(cif_path))
    return structure


def get_first_model(structure):
    return next(structure.get_models())


def sort_entity_ids(entity_ids: List[str]) -> List[str]:
    try:
        return sorted(entity_ids, key=lambda x: int(x))
    except Exception:
        return sorted(entity_ids)


def collect_entity_to_chains_from_mmcif_dict(cif_path: Path) -> Dict[str, List[str]]:
    """
    Build:
        entity_id -> ordered list of auth_asym_id chain IDs
    using atom-site information.
    """
    mmcif_dict = MMCIF2Dict(str(cif_path))

    entity_ids = mmcif_get_list(mmcif_dict, "_atom_site.label_entity_id")
    auth_asym_ids = mmcif_get_list(mmcif_dict, "_atom_site.auth_asym_id")

    if entity_ids is None or auth_asym_ids is None:
        raise ValueError(
            f"Missing _atom_site.label_entity_id or _atom_site.auth_asym_id in {cif_path}"
        )

    if len(entity_ids) != len(auth_asym_ids):
        raise ValueError(
            f"Mismatched lengths in {cif_path}: "
            f"{len(entity_ids)} entity ids vs {len(auth_asym_ids)} chain ids"
        )

    entity_to_chains: Dict[str, List[str]] = {}
    for entity_id, chain_id in zip(entity_ids, auth_asym_ids):
        entity_id = str(entity_id).strip()
        chain_id = str(chain_id).strip()
        if not entity_id or not chain_id or chain_id in {".", "?"}:
            continue
        if entity_id not in entity_to_chains:
            entity_to_chains[entity_id] = []
        if chain_id not in entity_to_chains[entity_id]:
            entity_to_chains[entity_id].append(chain_id)

    return entity_to_chains


def collect_entity_to_description_from_mmcif_dict(cif_path: Path) -> Dict[str, str]:
    """
    Collect entity_id -> description, usually from _entity.pdbx_description.
    """
    mmcif_dict = MMCIF2Dict(str(cif_path))
    entity_ids = mmcif_get_list(mmcif_dict, "_entity.id")
    descriptions = mmcif_get_list(mmcif_dict, "_entity.pdbx_description")

    result: Dict[str, str] = {}
    if entity_ids is None or descriptions is None or len(entity_ids) != len(descriptions):
        return result

    for ent_id, desc in zip(entity_ids, descriptions):
        ent_id = str(ent_id).strip()
        desc = str(desc).strip()
        if ent_id and desc not in {"", ".", "?"}:
            desc = desc.replace("\n", " ").replace("\r", " ").strip()
            result[ent_id] = desc

    return result


def collect_entity_to_ec_from_mmcif_dict(cif_path: Path) -> Dict[str, str]:
    """
    Try:
      1) _entity.pdbx_ec
      2) _entity_poly.pdbx_ec
    """
    mmcif_dict = MMCIF2Dict(str(cif_path))
    result: Dict[str, str] = {}

    entity_ids = mmcif_get_list(mmcif_dict, "_entity.id")
    entity_ecs = mmcif_get_list(mmcif_dict, "_entity.pdbx_ec")
    if entity_ids is not None and entity_ecs is not None and len(entity_ids) == len(entity_ecs):
        for ent_id, ec in zip(entity_ids, entity_ecs):
            ent_id = str(ent_id).strip()
            ec = str(ec).strip()
            if ent_id and ec not in {"", ".", "?"}:
                result[ent_id] = ec

    if result:
        return result

    poly_entity_ids = mmcif_get_list(mmcif_dict, "_entity_poly.entity_id")
    poly_entity_ecs = mmcif_get_list(mmcif_dict, "_entity_poly.pdbx_ec")
    if poly_entity_ids is not None and poly_entity_ecs is not None and len(poly_entity_ids) == len(poly_entity_ecs):
        for ent_id, ec in zip(poly_entity_ids, poly_entity_ecs):
            ent_id = str(ent_id).strip()
            ec = str(ec).strip()
            if ent_id and ec not in {"", ".", "?"}:
                result[ent_id] = ec

    return result


def collect_entity_to_uniprot_from_mmcif_dict(cif_path: Path) -> Dict[str, str]:
    """
    Collect entity_id -> UniProt accession(s) from _struct_ref.

    Primary mmCIF columns:
      - _struct_ref.entity_id
      - _struct_ref.pdbx_db_accession
      - _struct_ref.db_name
    """
    mmcif_dict = MMCIF2Dict(str(cif_path))

    entity_ids = mmcif_get_list(mmcif_dict, "_struct_ref.entity_id")
    accessions = mmcif_get_list(mmcif_dict, "_struct_ref.pdbx_db_accession")
    db_names = mmcif_get_list(mmcif_dict, "_struct_ref.db_name")

    if entity_ids is None or accessions is None or len(entity_ids) != len(accessions):
        return {}

    db_name_ok = (
        db_names is not None
        and len(db_names) == len(entity_ids)
    )

    valid_db_names = {"UNP", "UNIPROT", "UNIPROTKB", "SWISSPROT"}

    grouped: Dict[str, List[str]] = {}
    for idx, (ent_id, acc) in enumerate(zip(entity_ids, accessions)):
        ent_id = str(ent_id).strip()
        acc = str(acc).strip()
        if not ent_id or not acc or acc in {".", "?"}:
            continue

        if db_name_ok:
            db_name = str(db_names[idx]).strip().upper()
            if db_name not in valid_db_names:
                continue

        if ent_id not in grouped:
            grouped[ent_id] = []
        if acc not in grouped[ent_id]:
            grouped[ent_id].append(acc)

    return {ent_id: ",".join(accs) for ent_id, accs in grouped.items()}


def collect_polymer_entity_ids(cif_path: Path) -> Set[str]:
    """
    Return entity IDs that are polymer entities.
    Best source: _entity_poly.entity_id
    Fallback: _entity.type == polymer
    """
    mmcif_dict = MMCIF2Dict(str(cif_path))

    poly_entity_ids = mmcif_get_list(mmcif_dict, "_entity_poly.entity_id")
    if poly_entity_ids is not None:
        return {str(x).strip() for x in poly_entity_ids if str(x).strip() not in {"", ".", "?"}}

    entity_ids = mmcif_get_list(mmcif_dict, "_entity.id")
    entity_types = mmcif_get_list(mmcif_dict, "_entity.type")

    result: Set[str] = set()
    if entity_ids is not None and entity_types is not None and len(entity_ids) == len(entity_types):
        for ent_id, ent_type in zip(entity_ids, entity_types):
            ent_id = str(ent_id).strip()
            ent_type = str(ent_type).strip().lower()
            if ent_id and ent_type == "polymer":
                result.add(ent_id)

    return result


def choose_first_chain_per_entity(entity_to_chains: Dict[str, List[str]]) -> Dict[str, str]:
    chosen = {}
    for entity_id, chains in entity_to_chains.items():
        if chains:
            chosen[entity_id] = chains[0]
    return chosen


class KeepSelectedChains(Select):
    def __init__(self, chains_to_keep: Set[str]):
        self.chains_to_keep = chains_to_keep

    def accept_chain(self, chain) -> bool:
        return chain.id in self.chains_to_keep

    def accept_residue(self, residue) -> bool:
        return 1

    def accept_atom(self, atom) -> bool:
        return 1


def save_filtered_mmcif_temp(structure, out_cif: Path, chains_to_keep: Set[str]) -> None:
    io = MMCIFIO()
    io.set_structure(structure)
    io.save(str(out_cif), select=KeepSelectedChains(chains_to_keep))


def prepend_comments_to_mmcif(
    cif_path: Path,
    chosen_polymer_by_entity: Dict[str, str],
    entity_to_description: Dict[str, str],
    entity_to_ec: Dict[str, str],
    entity_to_uniprot: Dict[str, str],
) -> None:
    """
    Prepend safe mmCIF comment lines beginning with '#'.
    Only polymer entities are listed in the summary.
    """
    original_text = cif_path.read_text(encoding="utf-8", errors="replace")

    header_lines = []
    header_lines.append("# Filtered to first chain per polymer entity")
    for entity_id in sort_entity_ids(list(chosen_polymer_by_entity.keys())):
        chain_id = chosen_polymer_by_entity[entity_id]
        desc = entity_to_description.get(entity_id, "NOT_AVAILABLE")
        ec = entity_to_ec.get(entity_id, "NOT_AVAILABLE")
        uniprot = entity_to_uniprot.get(entity_id, "NOT_AVAILABLE")
        header_lines.append(
            f"# MOL_ID: {entity_id}; MOLECULE: {desc}; CHAIN: {chain_id}; EC: {ec}; UNIPROT: {uniprot};"
        )
    header_lines.append("# NOTE: same-chain hetero residues, ions, ligands, and waters were retained.")
    header_lines.append("")

    new_text = "\n".join(header_lines) + original_text
    cif_path.write_text(new_text, encoding="utf-8")


def get_atom_serial(atom) -> int:
    serial = getattr(atom, "serial_number", None)
    if serial is None:
        return 0
    try:
        return int(serial)
    except Exception:
        return 0


def classify_record_name(residue) -> str:
    hetflag = residue.id[0]
    if hetflag.strip() == "":
        return "ATOM"
    return "HETATM"


def format_pdb_atom_line(
    record_name: str,
    serial: int,
    atom_name: str,
    altloc: str,
    resname: str,
    chain_id: str,
    resseq: int,
    icode: str,
    x: float,
    y: float,
    z: float,
    occupancy: float,
    bfactor: float,
    element: str,
    charge: str = "",
) -> str:
    atom_name = atom_name[:4]
    altloc = (altloc or " ")[:1]
    resname = (resname or "")[:3].rjust(3)
    chain_id = (chain_id or " ")[:1]
    icode = (icode or " ")[:1]
    element = (element or "").strip().upper()[:2].rjust(2)
    charge = (charge or "")[:2].rjust(2)

    if len(atom_name.strip()) < 4 and not atom_name[:1].isdigit():
        atom_field = f" {atom_name.strip():<3}"
    else:
        atom_field = f"{atom_name:<4}"

    return (
        f"{record_name:<6}"
        f"{serial:>5} "
        f"{atom_field}"
        f"{altloc}"
        f"{resname} "
        f"{chain_id}"
        f"{resseq:>4}"
        f"{icode}   "
        f"{x:>8.3f}"
        f"{y:>8.3f}"
        f"{z:>8.3f}"
        f"{occupancy:>6.2f}"
        f"{bfactor:>6.2f}"
        f"          "
        f"{element}"
        f"{charge}"
        "\n"
    )


def wrap_compnd_field(serial_num: int, key: str, value: str) -> List[str]:
    """
    Write simple wrapped PDB COMPND lines.
    """
    value = value.strip()
    full_text = f"{key}: {value};"
    max_payload_len = 59
    chunks = [full_text[i:i + max_payload_len] for i in range(0, len(full_text), max_payload_len)]

    lines = []
    for chunk in chunks:
        lines.append(f"COMPND {serial_num:>3} {chunk}\n")
        serial_num += 1
    return lines


def build_pdb_compnd_block(
    chosen_polymer_by_entity: Dict[str, str],
    entity_to_description: Dict[str, str],
    entity_to_ec: Dict[str, str],
    entity_to_uniprot: Dict[str, str],
) -> List[str]:
    """
    Build a clean COMPND block for the reduced PDB.
    Only polymer entities are listed.
    """
    lines: List[str] = []
    serial_num = 1

    for entity_id in sort_entity_ids(list(chosen_polymer_by_entity.keys())):
        chain_id = chosen_polymer_by_entity[entity_id]
        desc = entity_to_description.get(entity_id, "NOT_AVAILABLE")
        ec = entity_to_ec.get(entity_id)
        uniprot = entity_to_uniprot.get(entity_id)

        block_fields = [
            ("MOL_ID", entity_id),
            ("MOLECULE", desc),
            ("CHAIN", chain_id),
        ]
        if ec:
            block_fields.append(("EC", ec))
        if uniprot:
            block_fields.append(("UNIPROT", uniprot))

        for key, value in block_fields:
            new_lines = wrap_compnd_field(serial_num, key, value)
            lines.extend(new_lines)
            serial_num += len(new_lines)

    return lines


def save_filtered_pdb_preserve_serials(
    structure,
    out_pdb: Path,
    chains_to_keep: Set[str],
    chosen_polymer_by_entity: Dict[str, str],
    entity_to_description: Dict[str, str],
    entity_to_ec: Dict[str, str],
    entity_to_uniprot: Dict[str, str],
) -> None:
    """
    Custom PDB writer:
    - writes a COMPND-style block at the top
    - lists only polymer entities there
    - keeps selected chains only
    - preserves author residue numbering
    - preserves atom serial numbers when available
    """
    model = get_first_model(structure)

    with open(out_pdb, "w", encoding="utf-8") as fh:
        fh.write("HEADER    FILTERED STRUCTURE\n")
        fh.writelines(build_pdb_compnd_block(
            chosen_polymer_by_entity=chosen_polymer_by_entity,
            entity_to_description=entity_to_description,
            entity_to_ec=entity_to_ec,
            entity_to_uniprot=entity_to_uniprot,
        ))
        fh.write("REMARK Filtered to first chain per polymer entity\n")
        fh.write("REMARK Same-chain hetero residues, ions, ligands, and waters were retained\n")

        last_written_serial = None

        for chain in model:
            if chain.id not in chains_to_keep:
                continue

            chain_had_atoms = False
            last_resname = None
            last_resseq = None
            last_icode = None

            for residue in chain:
                _, resseq, icode = residue.id
                resname = residue.resname.strip()

                for atom in residue:
                    serial = get_atom_serial(atom)
                    if serial <= 0:
                        serial = 1 if last_written_serial is None else last_written_serial + 1

                    x, y, z = map(float, atom.coord)
                    occupancy = atom.occupancy if atom.occupancy is not None else 1.00
                    bfactor = atom.bfactor if atom.bfactor is not None else 0.00
                    altloc = atom.altloc if atom.altloc not in (None, "") else " "
                    atom_name = atom.fullname.strip() if getattr(atom, "fullname", None) else atom.name
                    element = getattr(atom, "element", "") or atom_name[:1]

                    record_name = classify_record_name(residue)

                    line = format_pdb_atom_line(
                        record_name=record_name,
                        serial=serial,
                        atom_name=atom_name,
                        altloc=altloc,
                        resname=resname,
                        chain_id=chain.id,
                        resseq=resseq,
                        icode=icode if icode != " " else " ",
                        x=x,
                        y=y,
                        z=z,
                        occupancy=float(occupancy),
                        bfactor=float(bfactor),
                        element=element,
                        charge="",
                    )
                    fh.write(line)

                    chain_had_atoms = True
                    last_written_serial = serial
                    last_resname = resname
                    last_resseq = resseq
                    last_icode = icode

            if chain_had_atoms:
                ter_serial = 1 if last_written_serial is None else last_written_serial + 1
                ter_line = (
                    f"TER   "
                    f"{ter_serial:>5}      "
                    f"{(last_resname or ''):>3} "
                    f"{(chain.id or ' ')[:1]}"
                    f"{(last_resseq if last_resseq is not None else 0):>4}"
                    f"{(last_icode if last_icode not in (None, '') else ' ')[:1]}"
                    f"\n"
                )
                fh.write(ter_line)
                last_written_serial = ter_serial

        fh.write("END\n")


def process_one_file(cif_path: Path, out_cif_dir: Path, out_pdb_dir: Path) -> Tuple[Path, Path]:
    structure = load_structure(cif_path)
    first_model = get_first_model(structure)

    parsed_chain_ids = [chain.id for chain in first_model]
    if not parsed_chain_ids:
        raise ValueError(f"No chains found in parsed structure: {cif_path}")

    entity_to_chains = collect_entity_to_chains_from_mmcif_dict(cif_path)
    entity_to_description = collect_entity_to_description_from_mmcif_dict(cif_path)
    entity_to_ec = collect_entity_to_ec_from_mmcif_dict(cif_path)
    entity_to_uniprot = collect_entity_to_uniprot_from_mmcif_dict(cif_path)
    polymer_entity_ids = collect_polymer_entity_ids(cif_path)

    chosen_by_entity = choose_first_chain_per_entity(entity_to_chains)

    # Only polymer entities should determine which representative chains to keep
    chosen_polymer_by_entity = {
        ent_id: chain_id
        for ent_id, chain_id in chosen_by_entity.items()
        if ent_id in polymer_entity_ids
    }

    chains_to_keep = set(chosen_polymer_by_entity.values())
    chains_to_keep = {c for c in chains_to_keep if c in parsed_chain_ids}

    if not chains_to_keep:
        raise ValueError(f"No selected polymer chains found in parsed structure for {cif_path}")

    # Keep only entries whose chosen chain actually exists
    chosen_polymer_by_entity = {
        ent_id: chain_id
        for ent_id, chain_id in chosen_polymer_by_entity.items()
        if chain_id in chains_to_keep
    }

    out_cif = out_cif_dir / cif_path.name
    out_pdb = out_pdb_dir / f"{cif_path.stem}.pdb"

    save_filtered_mmcif_temp(structure, out_cif, chains_to_keep)
    prepend_comments_to_mmcif(
        cif_path=out_cif,
        chosen_polymer_by_entity=chosen_polymer_by_entity,
        entity_to_description=entity_to_description,
        entity_to_ec=entity_to_ec,
        entity_to_uniprot=entity_to_uniprot,
    )

    save_filtered_pdb_preserve_serials(
        structure=structure,
        out_pdb=out_pdb,
        chains_to_keep=chains_to_keep,
        chosen_polymer_by_entity=chosen_polymer_by_entity,
        entity_to_description=entity_to_description,
        entity_to_ec=entity_to_ec,
        entity_to_uniprot=entity_to_uniprot,
    )

    log(f"Processed: {cif_path.name}")
    log(f"  Parsed chains: {parsed_chain_ids}")
    log(f"  Polymer entity IDs: {sorted(polymer_entity_ids)}")
    log(f"  Entity -> chains: {entity_to_chains}")
    log(f"  Entity -> description: {entity_to_description}")
    log(f"  Entity -> EC: {entity_to_ec}")
    log(f"  Entity -> UniProt: {entity_to_uniprot}")
    log(f"  Kept chains: {sorted(chains_to_keep)}")
    log(f"  Saved mmCIF: {out_cif}")
    log(f"  Saved PDB:   {out_pdb}")
    log("")

    return out_cif, out_pdb


def main() -> None:
    if not INPUT_DIR.exists():
        raise FileNotFoundError(f"Input directory does not exist: {INPUT_DIR}")

    OUTPUT_CIF_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_PDB_DIR.mkdir(parents=True, exist_ok=True)

    cif_files = find_cif_files(INPUT_DIR)
    if not cif_files:
        print(f"No .cif or .mmcif files found in: {INPUT_DIR}")
        return

    print(f"Found {len(cif_files)} input mmCIF files.")
    print(f"Input:       {INPUT_DIR}")
    print(f"Output CIF:  {OUTPUT_CIF_DIR}")
    print(f"Output PDB:  {OUTPUT_PDB_DIR}")
    print("")

    n_ok = 0
    n_fail = 0

    for cif_path in cif_files:
        out_cif = OUTPUT_CIF_DIR / cif_path.name
        out_pdb = OUTPUT_PDB_DIR / f"{cif_path.stem}.pdb"

        if not OVERWRITE and out_cif.exists() and out_pdb.exists():
            log(f"Skipping existing outputs for: {cif_path.name}")
            log("")
            continue

        try:
            process_one_file(cif_path, OUTPUT_CIF_DIR, OUTPUT_PDB_DIR)
            n_ok += 1
        except Exception as e:
            n_fail += 1
            print(f"FAILED: {cif_path.name}")
            print(f"  Reason: {e}")
            print("")

    print("Done.")
    print(f"Successful: {n_ok}")
    print(f"Failed:     {n_fail}")


if __name__ == "__main__":
    main()
