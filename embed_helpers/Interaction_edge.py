import os
import sys
from pathlib import Path
import subprocess

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from project_paths import resolve_embeddings_dir


RING_EXE = Path("/home/mechti/ring-4.0/out/bin/ring")
DIR_RESULTS = resolve_embeddings_dir(os.getenv("EMBEDDINGS_DIR"))


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


if __name__ == "__main__":
    # path_test_structure = "/home/mechti/Downloads/AF-P06965-F1-model_v6.cif"
    path_test_structure = "/media/Data/pinmymetal_sets/mahomes/train_set/job_0/1a0e__chain_A__EC_5.3.1.5.pdb"
    ring_create_results(DIR_RESULTS, path_test_structure)
