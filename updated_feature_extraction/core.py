from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import biotite.structure as struc
import numpy as np
from biotite.structure.io import load_structure

from .constants import (
    FEATURE_NAMES,
    FORMAL_RESIDUE_CHARGES,
    KYTE_DOOLITTLE_HYDROPATHY,
    MAX_RESIDUE_SASA,
    METAL_FORMAL_CHARGES,
    RAMA_BASINS_DEGREES,
    ROTAMER_CENTERS_DEGREES,
)
from .propka_support import PropkaRunResult, run_propka_for_structure

ResidueKey = tuple[str, int, str]


@dataclass(frozen=True)
class ResidueGeometry:
    key: ResidueKey
    resname: str
    chain_id: str
    resseq: int
    icode: str
    heavy_coords: np.ndarray
    ca_coord: np.ndarray


def default_feature_dict() -> dict[str, float]:
    features: dict[str, float] = {}
    for name in FEATURE_NAMES:
        features[name] = 0.0
        features[f"{name}_missing"] = 1.0
    return features


def _set_feature(entry: dict[str, float], name: str, value: float) -> None:
    entry[name] = float(value)
    missing_name = f"{name}_missing"
    if missing_name in entry:
        entry[missing_name] = 0.0


def _angular_distance_degrees(angle: float, reference: float) -> float:
    return abs(((angle - reference + 180.0) % 360.0) - 180.0)


def _nearest_rotamer_deviation(angle_deg: float) -> float:
    return min(_angular_distance_degrees(angle_deg, center) for center in ROTAMER_CENTERS_DEGREES) / 60.0


def _rama_basins_for_residue(resname: str, next_resname: str | None) -> list[tuple[float, float]]:
    if resname == "GLY":
        return RAMA_BASINS_DEGREES["gly"]
    if resname == "PRO":
        return RAMA_BASINS_DEGREES["pro"]
    if next_resname == "PRO":
        return RAMA_BASINS_DEGREES["prepro"]
    return RAMA_BASINS_DEGREES["general"]


def _residue_identifier(residue) -> ResidueKey:
    return (
        str(residue.chain_id[0]),
        int(residue.res_id[0]),
        str(residue.ins_code[0]).strip(),
    )


def _residue_geometry_from_atom_array(atom_array) -> list[ResidueGeometry]:
    residues: list[ResidueGeometry] = []
    for residue in struc.residue_iter(atom_array):
        resname = str(residue.res_name[0]).strip()
        coords = residue.coord.astype(np.float64, copy=False)
        heavy_mask = residue.element != "H"
        heavy_coords = coords[heavy_mask]
        if heavy_coords.size == 0:
            continue
        ca_indices = np.where(residue.atom_name == "CA")[0]
        if ca_indices.size == 0:
            continue
        residues.append(
            ResidueGeometry(
                key=_residue_identifier(residue),
                resname=resname,
                chain_id=str(residue.chain_id[0]),
                resseq=int(residue.res_id[0]),
                icode=str(residue.ins_code[0]).strip(),
                heavy_coords=heavy_coords,
                ca_coord=coords[int(ca_indices[0])],
            )
        )
    return residues


def _minimum_pair_distance(coords_a: np.ndarray, coords_b: np.ndarray) -> float:
    deltas = coords_a[:, np.newaxis, :] - coords_b[np.newaxis, :, :]
    return float(np.linalg.norm(deltas, axis=-1).min())


def _iter_metal_sites(atom_array) -> Iterable[tuple[str, np.ndarray]]:
    hetero_metals = atom_array[
        np.isin(atom_array.element, np.array(list(METAL_FORMAL_CHARGES)))
    ]
    for atom in hetero_metals:
        yield str(atom.element), atom.coord.astype(np.float64, copy=False)


def _apply_sasa_and_burial(
    atom_array,
    residues: list[ResidueGeometry],
    feature_map: dict[ResidueKey, dict[str, float]],
) -> None:
    protein_mask = struc.filter_canonical_amino_acids(atom_array)
    protein_atoms = atom_array[protein_mask]
    sasa = struc.sasa(
        protein_atoms,
        ignore_ions=True,
    )
    atom_starts = struc.get_residue_starts(protein_atoms)
    atom_stops = list(atom_starts[1:]) + [protein_atoms.array_length()]

    for residue, start, stop in zip(residues, atom_starts, atom_stops):
        residue_sasa = float(np.nansum(sasa[start:stop]))
        max_sasa = MAX_RESIDUE_SASA.get(residue.resname, max(MAX_RESIDUE_SASA.values()))
        buried_surface = max(0.0, max_sasa - residue_sasa)
        burial_fraction = min(1.0, buried_surface / max_sasa) if max_sasa > 0.0 else 0.0
        hydropathy = KYTE_DOOLITTLE_HYDROPATHY.get(residue.resname, 0.0)

        _set_feature(feature_map[residue.key], "SASA", residue_sasa)
        _set_feature(feature_map[residue.key], "BSA", buried_surface)
        _set_feature(feature_map[residue.key], "fa_sol", burial_fraction)
        # Negative values indicate favorable burial of hydrophobics, while
        # exposed or buried polar residues drift toward neutral/positive.
        _set_feature(feature_map[residue.key], "SolvEnergy", -hydropathy * burial_fraction)


def _apply_backbone_and_rotamer_scores(
    atom_array,
    residues: list[ResidueGeometry],
    feature_map: dict[ResidueKey, dict[str, float]],
) -> None:
    phi, psi, omega = struc.dihedral_backbone(atom_array)
    chi = struc.dihedral_side_chain(atom_array)
    residue_names = [residue.resname for residue in residues]

    for index, residue in enumerate(residues):
        omega_deg = math.degrees(float(omega[index])) if not np.isnan(omega[index]) else 180.0
        omega_score = min(
            _angular_distance_degrees(omega_deg, 180.0),
            _angular_distance_degrees(omega_deg, -180.0),
            _angular_distance_degrees(omega_deg, 0.0),
        ) / 180.0

        next_resname = residue_names[index + 1] if index + 1 < len(residue_names) else None
        phi_deg = math.degrees(float(phi[index])) if not np.isnan(phi[index]) else None
        psi_deg = math.degrees(float(psi[index])) if not np.isnan(psi[index]) else None
        if phi_deg is None or psi_deg is None:
            rama_score = 0.0
        else:
            rama_score = min(
                math.sqrt(
                    _angular_distance_degrees(phi_deg, basin_phi) ** 2
                    + _angular_distance_degrees(psi_deg, basin_psi) ** 2
                )
                for basin_phi, basin_psi in _rama_basins_for_residue(residue.resname, next_resname)
            ) / 180.0

        chi_angles = chi[index]
        valid_chi = [math.degrees(float(value)) for value in chi_angles if not np.isnan(value)]
        if valid_chi:
            dun_score = sum(_nearest_rotamer_deviation(angle) for angle in valid_chi) / len(valid_chi)
        else:
            dun_score = 0.0

        _set_feature(feature_map[residue.key], "omega", omega_score)
        _set_feature(feature_map[residue.key], "rama_prepro", rama_score)
        _set_feature(feature_map[residue.key], "fa_dun", dun_score)


def _apply_pairwise_interaction_proxies(
    residues: list[ResidueGeometry],
    atom_array,
    feature_map: dict[ResidueKey, dict[str, float]],
) -> None:
    for residue in residues:
        _set_feature(feature_map[residue.key], "fa_elec", feature_map[residue.key]["fa_elec"])
        _set_feature(feature_map[residue.key], "fa_atr", feature_map[residue.key]["fa_atr"])
        _set_feature(feature_map[residue.key], "fa_rep", feature_map[residue.key]["fa_rep"])

    for left_index, residue_left in enumerate(residues):
        for right_index in range(left_index + 1, len(residues)):
            residue_right = residues[right_index]
            if (
                residue_left.chain_id == residue_right.chain_id
                and abs(residue_left.resseq - residue_right.resseq) <= 1
            ):
                continue
            ca_distance = float(np.linalg.norm(residue_left.ca_coord - residue_right.ca_coord))
            if ca_distance > 12.0:
                continue

            min_distance = _minimum_pair_distance(residue_left.heavy_coords, residue_right.heavy_coords)
            if min_distance <= 0.0:
                continue

            attraction = 0.0
            repulsion = 0.0
            electrostatics = 0.0

            if min_distance < 2.6:
                repulsion = (2.6 - min_distance) ** 2
            elif min_distance < 4.5:
                attraction = -((4.5 - min_distance) ** 2) / 4.0

            charge_left = FORMAL_RESIDUE_CHARGES.get(residue_left.resname, 0.0)
            charge_right = FORMAL_RESIDUE_CHARGES.get(residue_right.resname, 0.0)
            if charge_left and charge_right:
                electrostatics = (charge_left * charge_right) / max(min_distance, 2.5)

            for key in (residue_left.key, residue_right.key):
                _set_feature(feature_map[key], "fa_atr", feature_map[key]["fa_atr"] + attraction)
                _set_feature(feature_map[key], "fa_rep", feature_map[key]["fa_rep"] + repulsion)
                _set_feature(feature_map[key], "fa_elec", feature_map[key]["fa_elec"] + electrostatics)

    metal_sites = list(_iter_metal_sites(atom_array))
    if not metal_sites:
        return

    for residue in residues:
        charge = FORMAL_RESIDUE_CHARGES.get(residue.resname, 0.0)
        if not charge:
            continue
        for metal_element, metal_coord in metal_sites:
            min_distance = float(np.linalg.norm(residue.heavy_coords - metal_coord, axis=1).min())
            metal_charge = METAL_FORMAL_CHARGES.get(metal_element, 0.0)
            if not metal_charge:
                continue
            _set_feature(
                feature_map[residue.key],
                "fa_elec",
                feature_map[residue.key]["fa_elec"] + (charge * metal_charge) / max(min_distance, 2.0),
            )


def _apply_propka_features(
    residues: list[ResidueGeometry],
    feature_map: dict[ResidueKey, dict[str, float]],
    propka_result: PropkaRunResult | None,
) -> list[str]:
    if propka_result is None:
        return []

    residue_index_by_chain_resseq_resname: dict[tuple[str, int, str], list[ResidueKey]] = {}
    for residue in residues:
        residue_index_by_chain_resseq_resname.setdefault(
            (residue.chain_id, residue.resseq, residue.resname),
            [],
        ).append(residue.key)

    for compact_key, propka_features in propka_result.residues.items():
        chain_id, resseq, resname = compact_key
        for residue_key in residue_index_by_chain_resseq_resname.get((chain_id, resseq, resname), []):
            entry = feature_map[residue_key]
            _set_feature(entry, "pKa_shift", propka_features.model_pka - propka_features.predicted_pka)
            _set_feature(entry, "dpKa_desolv", propka_features.dpka_desolv)
            _set_feature(entry, "dpKa_bg", propka_features.dpka_bg)
            _set_feature(entry, "dpKa_titr", propka_features.dpka_titr)
            _set_feature(entry, "fa_elec", entry["fa_elec"] + propka_features.dpka_titr)
    return propka_result.warnings


def generate_feature_map_for_structure(
    structure_path: str | Path,
    *,
    propka_ph: float = 7.0,
    include_propka: bool = True,
) -> tuple[dict[ResidueKey, dict[str, float]], dict[str, object]]:
    structure_path = Path(structure_path)
    atom_array = load_structure(str(structure_path))
    protein_atoms = atom_array[struc.filter_canonical_amino_acids(atom_array)]
    residues = _residue_geometry_from_atom_array(protein_atoms)
    feature_map = {residue.key: default_feature_dict() for residue in residues}

    _apply_sasa_and_burial(atom_array, residues, feature_map)
    _apply_backbone_and_rotamer_scores(protein_atoms, residues, feature_map)
    _apply_pairwise_interaction_proxies(residues, atom_array, feature_map)

    warnings: list[str] = []
    propka_used = False
    if include_propka:
        try:
            propka_result = run_propka_for_structure(structure_path, ph=propka_ph)
        except Exception as exc:
            warnings.append(str(exc))
            propka_result = None
        else:
            propka_used = True
        warnings.extend(_apply_propka_features(residues, feature_map, propka_result))

    metadata = {
        "structure_id": structure_path.stem,
        "source_path": str(structure_path),
        "feature_names": list(FEATURE_NAMES),
        "tooling": {
            "geometry": "biotite",
            "pka": "propka" if propka_used else "unavailable",
        },
        "warnings": warnings,
        "n_residues": len(residues),
    }
    return feature_map, metadata


def build_structure_feature_payload(
    structure_path: str | Path,
    *,
    propka_ph: float = 7.0,
    include_propka: bool = True,
) -> dict[str, object]:
    feature_map, metadata = generate_feature_map_for_structure(
        structure_path,
        propka_ph=propka_ph,
        include_propka=include_propka,
    )
    residues_payload = [
        {
            "chain_id": chain_id,
            "resseq": resseq,
            "icode": icode,
            "features": feature_map[(chain_id, resseq, icode)],
        }
        for chain_id, resseq, icode in sorted(feature_map)
    ]
    return {
        "schema_version": 1,
        **metadata,
        "residues": residues_payload,
    }
