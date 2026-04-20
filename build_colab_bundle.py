from __future__ import annotations

"""Build a portable, Colab-ready dataset bundle from the local DeepGM dataset.

This script does not change the training code and it does not train a model.
Its job is to prepare data so the existing training path can be moved to
another machine, especially Colab, without hand-copying folders.

High-level flow:

1. Read the training tree and the summary CSV that define the usable dataset.
2. Validate structures with the same loader path used by training.
3. Classify each structure as:
   - included: loader succeeded and at least one pocket is kept
   - unused: loader succeeded but nothing remains after filtering
   - invalid: loader failed for that structure
4. Write a manifest JSON that records what was included, unused, invalid, and
   explicitly excluded.
5. Optionally build two archives:
   - a train-set archive that preserves the folder layout expected by training
   - an embeddings archive containing the matching ESM files

There are two operating modes:

- Fresh scan mode:
  validate the dataset, write a manifest, and optionally archive the results.
- Manifest reuse mode:
  skip the expensive validation scan, load a previously written manifest, and
  archive exactly the structures listed there.

This keeps the core DeepGM runtime local-machine friendly and Colab support
limited to packaging and wrapper tooling.
"""

import argparse
import json
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

from project_paths import EMBEDDINGS_DIR
from training.config import VALID_UNSUPPORTED_METAL_POLICY_CHOICES
from training.data import DEFAULT_TRAIN_SUMMARY_CSV
from training.defaults import DEFAULT_STRUCTURE_DIR
from training.esm_feature_loading import DEFAULT_ESMC_EMBED_DIM, embedding_path_candidates
from training.feature_paths import resolve_runtime_feature_paths
from training.feature_sources import resolve_structure_feature_dir
from training.site_filter import resolve_allowed_site_metal_labels
from training.structure_loading import find_structure_files, load_structure_pockets


@dataclass(frozen=True)
class StructureCheckResult:
    """One structure-level decision from the validation pass."""

    structure_id: str
    relative_structure_path: str
    status: str
    kept_pockets: int
    feature_fallbacks: list[dict[str, str]]
    skipped_pockets: list[dict[str, str]]
    error: str | None = None


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate the training dataset with the current loaders and build Colab-ready archives. "
            "The training code itself remains unchanged; this is packaging tooling around it."
        )
    )
    parser.add_argument("--structure-dir", type=Path, default=DEFAULT_STRUCTURE_DIR)
    parser.add_argument("--summary-csv", type=Path, default=DEFAULT_TRAIN_SUMMARY_CSV)
    parser.add_argument("--esm-embeddings-dir", type=Path, default=EMBEDDINGS_DIR)
    parser.add_argument("--external-features-root-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=None,
        help="Reuse an existing manifest instead of rescanning the dataset.",
    )
    parser.add_argument("--manifest-name", type=str, default="colab_bundle_manifest.json")
    parser.add_argument("--structures-archive-name", type=str, default="train_set_clean.tar.zst")
    parser.add_argument("--embeddings-archive-name", type=str, default="embeddings_clean.tar.zst")
    parser.add_argument("--esm-dim", type=int, default=DEFAULT_ESMC_EMBED_DIM)
    parser.add_argument(
        "--unsupported-metal-policy",
        type=str,
        default="error",
        choices=VALID_UNSUPPORTED_METAL_POLICY_CHOICES,
    )
    parser.add_argument("--allow-missing-esm-embeddings", action="store_true")
    parser.add_argument("--allow-missing-external-features", action="store_true")
    parser.add_argument("--exclude-structure-id", action="append", default=[])
    parser.add_argument("--manifest-only", action="store_true")
    return parser


def choose_tar_flags(archive_path: Path) -> list[str]:
    # Keep archive format selection explicit so the rest of the script can stay
    # format-agnostic.
    name = archive_path.name
    if name.endswith(".tar.zst"):
        return ["--zstd", "-cf"]
    if name.endswith(".tar.gz") or name.endswith(".tgz"):
        return ["-czf"]
    if name.endswith(".tar"):
        return ["-cf"]
    raise ValueError(f"Unsupported archive name {archive_path}. Use .tar, .tar.gz, or .tar.zst")


def to_relative_paths(base_dir: Path, members: Sequence[Path]) -> list[str]:
    # `tar -C <base>` expects members relative to that base directory.
    unique_members = sorted({member.resolve() for member in members})
    return [str(member.relative_to(base_dir)) for member in unique_members]


def select_embedding_members(embeddings_dir: Path, structure_path: Path) -> list[Path]:
    # Reuse the runtime candidate logic so the bundle contains exactly the
    # embedding files the loader would search for.
    return [candidate for candidate in embedding_path_candidates(embeddings_dir, structure_path) if candidate.is_file()]


def select_structure_members(
    *,
    structure_path: Path,
    structure_root: Path,
    feature_root_dir: Path | None,
    summary_csv: Path,
) -> list[Path]:
    # The structure archive needs more than the top-level `.pdb` file:
    # training also expects the per-structure feature directory and the summary
    # table directory that defines the catalytic filtering.
    members: list[Path] = [structure_path]
    feature_dir = resolve_structure_feature_dir(
        structure_path=structure_path,
        structure_root=structure_root,
        feature_root_dir=feature_root_dir,
    )
    if feature_dir is not None:
        members.append(feature_dir)

    summary_parent = summary_csv.parent
    if summary_parent.is_dir():
        members.append(summary_parent)
    elif summary_csv.is_file():
        members.append(summary_csv)
    return members


def validate_structure(
    *,
    structure_path: Path,
    structure_root: Path,
    allowed_site_metal_labels: dict[tuple[str, str, str], str] | None,
    embeddings_dir: Path,
    feature_root_dir: Path | None,
    esm_dim: int,
    require_esm_embeddings: bool,
    require_external_features: bool,
    unsupported_metal_policy: str,
) -> StructureCheckResult:
    # This is the key design choice of the script: validate with the same
    # loader path used by training, so the bundle is based on real runtime
    # behavior rather than filename heuristics.
    try:
        pockets, feature_fallbacks, skipped_pockets = load_structure_pockets(
            structure_path=structure_path,
            structure_root=structure_root,
            allowed_site_metal_labels=allowed_site_metal_labels,
            esm_dim=esm_dim,
            embeddings_dir=embeddings_dir,
            require_esm_embeddings=require_esm_embeddings,
            feature_root_dir=feature_root_dir or structure_root,
            require_external_features=require_external_features,
            unsupported_metal_policy=unsupported_metal_policy,
        )
    except Exception as exc:
        return StructureCheckResult(
            structure_id=structure_path.stem,
            relative_structure_path=str(structure_path.relative_to(structure_root)),
            status="invalid",
            kept_pockets=0,
            feature_fallbacks=[],
            skipped_pockets=[],
            error=str(exc),
        )

    status = "included" if pockets else "unused"
    return StructureCheckResult(
        structure_id=structure_path.stem,
        relative_structure_path=str(structure_path.relative_to(structure_root)),
        status=status,
        kept_pockets=len(pockets),
        feature_fallbacks=feature_fallbacks,
        skipped_pockets=skipped_pockets,
        error=None,
    )


def archive_members(archive_path: Path, base_dir: Path, members: Sequence[Path]) -> None:
    # We hand `tar` a file list instead of a huge argument vector so this still
    # works when the included dataset is large.
    relative_members = to_relative_paths(base_dir, members)
    if not relative_members:
        raise ValueError(f"No archive members selected for {archive_path}")

    archive_path.parent.mkdir(parents=True, exist_ok=True)
    tar_flags = choose_tar_flags(archive_path)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False) as handle:
        filelist_path = Path(handle.name)
        handle.write("\n".join(relative_members))
        handle.write("\n")

    try:
        subprocess.run(
            ["tar", "--dereference", *tar_flags, str(archive_path), "-C", str(base_dir), "-T", str(filelist_path)],
            check=True,
        )
    finally:
        filelist_path.unlink(missing_ok=True)


def build_manifest_payload(
    *,
    structure_dir: Path,
    summary_csv: Path,
    embeddings_dir: Path,
    feature_root_dir: Path | None,
    results: Sequence[StructureCheckResult],
    excluded_structure_ids: Sequence[str],
    structure_archive_name: str | None,
    embeddings_archive_name: str | None,
) -> dict[str, object]:
    # The manifest is the durable report of what the validation pass decided.
    # It is intentionally verbose so later packaging runs can reuse it and so
    # the user can inspect what was included or rejected.
    included = [result for result in results if result.status == "included"]
    invalid = [result for result in results if result.status == "invalid"]
    unused = [result for result in results if result.status == "unused"]

    return {
        "structure_dir": str(structure_dir),
        "summary_csv": str(summary_csv),
        "esm_embeddings_dir": str(embeddings_dir),
        "external_features_root_dir": str(feature_root_dir) if feature_root_dir is not None else None,
        "excluded_structure_ids": sorted(excluded_structure_ids),
        "n_total_structures": len(results) + len(excluded_structure_ids),
        "n_included_structures": len(included),
        "n_unused_structures": len(unused),
        "n_invalid_structures": len(invalid),
        "structure_archive_name": structure_archive_name,
        "embeddings_archive_name": embeddings_archive_name,
        "included_structures": [asdict(result) for result in included],
        "unused_structures": [asdict(result) for result in unused],
        "invalid_structures": [asdict(result) for result in invalid],
    }


def load_manifest_payload(manifest_path: Path) -> dict[str, object]:
    """Load a previously generated manifest for archive-only runs."""

    return json.loads(manifest_path.read_text(encoding="utf-8"))


def collect_members_from_manifest(
    *,
    manifest_payload: dict[str, object],
    structure_dir: Path,
    summary_csv: Path,
    embeddings_dir: Path,
    feature_root_dir: Path | None,
) -> tuple[list[Path], list[Path]]:
    # Manifest reuse mode avoids rescanning the entire dataset. We trust the
    # earlier validation result and reconstruct only the archive member lists.
    structure_members: list[Path] = []
    embedding_members: list[Path] = []

    for item in manifest_payload.get("included_structures", []):
        relative_structure_path = item["relative_structure_path"]
        structure_path = structure_dir / relative_structure_path
        structure_members.extend(
            select_structure_members(
                structure_path=structure_path,
                structure_root=structure_dir,
                feature_root_dir=feature_root_dir,
                summary_csv=summary_csv,
            )
        )
        embedding_members.extend(select_embedding_members(embeddings_dir, structure_path))

    return structure_members, embedding_members


def print_next_steps(
    *,
    structure_archive_path: Path | None,
    embeddings_archive_path: Path | None,
) -> None:
    # Print runnable Colab commands at the end so the bundle artifact is
    # immediately usable without extra translation.
    def unpack_command_for(archive_path: Path) -> str:
        if archive_path.name.endswith(".tar.zst"):
            return f"!tar --zstd -xf {archive_path.name} -C /content/DeepGM_data"
        if archive_path.name.endswith(".tar.gz") or archive_path.name.endswith(".tgz"):
            return f"!tar -xzf {archive_path.name} -C /content/DeepGM_data"
        return f"!tar -xf {archive_path.name} -C /content/DeepGM_data"

    if structure_archive_path is not None:
        print("Suggested Colab unpack command for structures:")
        print(f"  {unpack_command_for(structure_archive_path)}")
    if embeddings_archive_path is not None:
        print("Suggested Colab unpack command for embeddings:")
        print(f"  {unpack_command_for(embeddings_archive_path)}")
    print("Suggested Colab training command:")
    print(
        "  !python deepgm_colab.py "
        "--structure-dir /content/DeepGM_data/train_set "
        "--summary-csv /content/DeepGM_data/train_set/data_summarizing_tables/"
        "final_data_summarazing_table_transition_metals_only_catalytic.csv "
        "--esm-embeddings-dir /content/DeepGM_data/embeddings "
        "--runs-dir /content/drive/MyDrive/DeepGM/training_runs "
        "--device cuda"
    )


def main(argv: Sequence[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = args.manifest_path.expanduser().resolve() if args.manifest_path is not None else None
    if manifest_path is not None:
        # Fast path: skip validation and package from a known manifest.
        manifest_payload = load_manifest_payload(manifest_path)
        structure_dir = Path(str(manifest_payload["structure_dir"])).expanduser().resolve()
        summary_csv = Path(str(manifest_payload["summary_csv"])).expanduser().resolve()
        resolved_embeddings_dir = Path(str(manifest_payload["esm_embeddings_dir"])).expanduser().resolve()
        manifest_feature_root_dir = (
            Path(str(manifest_payload["external_features_root_dir"])).expanduser().resolve()
            if manifest_payload.get("external_features_root_dir")
            else None
        )
        _unused_embeddings_dir, resolved_feature_root_dir = resolve_runtime_feature_paths(
            structure_dir=structure_dir,
            esm_embeddings_dir=resolved_embeddings_dir,
            external_features_root_dir=(
                args.external_features_root_dir.expanduser().resolve()
                if args.external_features_root_dir is not None
                else manifest_feature_root_dir
            ),
        )
        structure_members, embedding_members = collect_members_from_manifest(
            manifest_payload=manifest_payload,
            structure_dir=structure_dir,
            summary_csv=summary_csv,
            embeddings_dir=resolved_embeddings_dir,
            feature_root_dir=resolved_feature_root_dir,
        )
        print(f"Loaded manifest from {manifest_path}")
    else:
        # Full path: validate every discoverable structure, decide whether it is
        # included/unused/invalid, and collect only the members needed for the
        # final bundle.
        structure_dir = args.structure_dir.expanduser().resolve()
        summary_csv = args.summary_csv.expanduser().resolve()
        embeddings_dir = args.esm_embeddings_dir.expanduser().resolve()
        feature_root_dir = (
            args.external_features_root_dir.expanduser().resolve() if args.external_features_root_dir else None
        )
        excluded_structure_ids = set(args.exclude_structure_id)
        require_esm_embeddings = not args.allow_missing_esm_embeddings
        require_external_features = not args.allow_missing_external_features
        allowed_site_metal_labels = resolve_allowed_site_metal_labels(summary_csv)
        resolved_embeddings_dir, resolved_feature_root_dir = resolve_runtime_feature_paths(
            structure_dir=structure_dir,
            esm_embeddings_dir=embeddings_dir,
            external_features_root_dir=feature_root_dir,
        )

        results: list[StructureCheckResult] = []
        structure_members = []
        embedding_members = []

        for structure_path in find_structure_files(structure_dir):
            # Explicit exclusions are a pragmatic escape hatch for known bad
            # structures that should not block the whole bundle build.
            if structure_path.stem in excluded_structure_ids:
                continue

            result = validate_structure(
                structure_path=structure_path,
                structure_root=structure_dir,
                allowed_site_metal_labels=allowed_site_metal_labels,
                embeddings_dir=resolved_embeddings_dir,
                feature_root_dir=resolved_feature_root_dir,
                esm_dim=args.esm_dim,
                require_esm_embeddings=require_esm_embeddings,
                require_external_features=require_external_features,
                unsupported_metal_policy=args.unsupported_metal_policy,
            )
            results.append(result)
            if result.status != "included":
                continue

            # Only included structures contribute files to the portable bundle.
            structure_members.extend(
                select_structure_members(
                    structure_path=structure_path,
                    structure_root=structure_dir,
                    feature_root_dir=resolved_feature_root_dir,
                    summary_csv=summary_csv,
                )
            )
            embedding_members.extend(select_embedding_members(resolved_embeddings_dir, structure_path))

        structure_archive_name = None if args.manifest_only else args.structures_archive_name
        embeddings_archive_name = None if args.manifest_only else args.embeddings_archive_name
        manifest_payload = build_manifest_payload(
            structure_dir=structure_dir,
            summary_csv=summary_csv,
            embeddings_dir=resolved_embeddings_dir,
            feature_root_dir=resolved_feature_root_dir,
            results=results,
            excluded_structure_ids=sorted(excluded_structure_ids),
            structure_archive_name=structure_archive_name,
            embeddings_archive_name=embeddings_archive_name,
        )
        manifest_path = output_dir / args.manifest_name
        manifest_path.write_text(json.dumps(manifest_payload, indent=2), encoding="utf-8")
        print(f"Wrote manifest to {manifest_path}")

    if args.manifest_only:
        # Stop after writing the report when the user only wants validation.
        return

    structure_archive_path = output_dir / args.structures_archive_name
    embeddings_archive_path = output_dir / args.embeddings_archive_name

    # The two archives are split intentionally: structure/features and ESM
    # embeddings can be stored, transferred, or regenerated independently.
    archive_members(structure_archive_path, structure_dir.parent, structure_members)
    archive_members(embeddings_archive_path, resolved_embeddings_dir.parent, embedding_members)
    print(f"Wrote structure archive to {structure_archive_path}")
    print(f"Wrote embeddings archive to {embeddings_archive_path}")
    print_next_steps(
        structure_archive_path=structure_archive_path,
        embeddings_archive_path=embeddings_archive_path,
    )


if __name__ == "__main__":
    main()
