import os
import sys
from pathlib import Path
import subprocess
from typing import Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from project_paths import resolve_embeddings_dir


RING_EXE = Path("/home/mechti/ring-4.0/out/bin/ring")
DIR_RESULTS = resolve_embeddings_dir(os.getenv("EMBEDDINGS_DIR"))


def expected_ring_edges_path(dir_results, path_structure) -> Path:
    dir_results = Path(dir_results)
    path_structure = Path(path_structure)
    return dir_results / path_structure.stem / f"{path_structure.name}_ringEdges"


def ring_create_results(dir_results, path_structure):
    dir_results = Path(dir_results)
    path_structure = Path(path_structure)

    dir_results.mkdir(parents=True, exist_ok=True)

    dir_structure_results = dir_results / path_structure.stem
    dir_structure_results.mkdir(parents=True, exist_ok=True)

    subprocess.run(
        [
            str(RING_EXE),
            "-i",
            str(path_structure),
            "--out_dir",
            str(dir_structure_results),
        ],
        check=True
    )
    return expected_ring_edges_path(dir_results, path_structure)


def create_ring_edges_batch(
    structure_files: Sequence[str | Path],
    dir_results: str | Path | None = None,
    *,
    overwrite: bool = False,
) -> dict[str, object]:
    if dir_results is None:
        dir_results = DIR_RESULTS
    dir_results = Path(dir_results)
    dir_results.mkdir(parents=True, exist_ok=True)

    if not RING_EXE.is_file():
        raise FileNotFoundError(f"RING executable not found: {RING_EXE}")

    processed = 0
    failed: list[dict[str, str]] = []
    saved_files: list[str] = []

    for structure_file in structure_files:
        structure_path = Path(structure_file)
        out_file = expected_ring_edges_path(dir_results, structure_path)
        if out_file.is_file() and not overwrite:
            saved_files.append(str(out_file))
            continue
        try:
            saved_path = ring_create_results(dir_results, structure_path)
            processed += 1
            saved_files.append(str(saved_path))
        except Exception as exc:
            failed.append(
                {
                    "structure_file": str(structure_path),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )

    return {
        "processed_structures": processed,
        "failed_structures": failed,
        "saved_files": saved_files,
        "out_dir": str(dir_results),
    }


if __name__ == "__main__":
    # path_test_structure = "/home/mechti/Downloads/AF-P06965-F1-model_v6.cif"
    path_test_structure = "/media/Data/pinmymetal_sets/mahomes/train_set/job_0/1a0e__chain_A__EC_5.3.1.5.pdb"
    ring_create_results(DIR_RESULTS, path_test_structure)
