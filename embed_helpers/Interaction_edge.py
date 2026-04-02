from pathlib import Path
import subprocess

RING_EXE = Path("/home/mechti/ring-4.0/out/bin/ring")
DIR_RESULTS = Path("/home/mechti/ring_results")


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
    path_test_structure = "/home/mechti/Downloads/AF-P06965-F1-model_v6.cif"
    ring_create_results(DIR_RESULTS, path_test_structure)