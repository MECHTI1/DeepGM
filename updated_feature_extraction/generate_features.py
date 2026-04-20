from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from project_paths import DATA_DIR
from training.structure_loading import find_structure_files
from updated_feature_extraction.core import build_structure_feature_payload


DEFAULT_OUTPUT_ROOT = DATA_DIR / "updated_feature_extraction"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate DeepGM external residue features with Biotite/PROPKA and "
            "save them into a .data-style structure-indexed directory tree."
        )
    )
    parser.add_argument("--structure-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--propka-ph", type=float, default=7.0)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--disable-propka", action="store_true")
    return parser


def write_structure_payload(output_root: Path, structure_path: Path, payload: dict[str, object]) -> Path:
    structure_dir = output_root / structure_path.stem
    structure_dir.mkdir(parents=True, exist_ok=True)
    output_path = structure_dir / "residue_features.json"
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output_path


def main(argv: Sequence[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    structure_files = find_structure_files(args.structure_dir)
    if not structure_files:
        raise FileNotFoundError(f"No structure files found under {args.structure_dir}")

    args.output_root.mkdir(parents=True, exist_ok=True)
    saved_files = 0
    skipped_files = 0
    failed_structures: list[dict[str, str]] = []

    for structure_path in structure_files:
        output_path = args.output_root / structure_path.stem / "residue_features.json"
        if args.skip_existing and output_path.is_file():
            skipped_files += 1
            continue

        try:
            payload = build_structure_feature_payload(
                structure_path,
                propka_ph=args.propka_ph,
                include_propka=not args.disable_propka,
            )
            write_structure_payload(args.output_root, structure_path, payload)
        except Exception as exc:
            failure = {
                "structure_path": str(structure_path),
                "error": str(exc),
            }
            failed_structures.append(failure)
            print(f"[failed] {structure_path}: {exc}")
            continue

        saved_files += 1
        print(f"[saved] {output_path}")

    failures_path = args.output_root / "generation_failures.json"
    failures_path.write_text(json.dumps(failed_structures, indent=2), encoding="utf-8")

    print(
        f"Generated updated external features for {saved_files} structures "
        f"(skipped {skipped_files}, failed {len(failed_structures)}) into {args.output_root}"
    )


if __name__ == "__main__":
    main()
