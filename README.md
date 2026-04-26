# DeepGM
Testing different variant:
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MECHTI1/DeepGM/blob/main/notebooks/run_deepgm.ipynb)
Use the project interpreter `/home/mechti/miniconda3/envs/deepgm-py312/bin/python`.
Before running project code, verify:
- `which python`
- `python -c "import sys; print(sys.executable)"`
If either command does not resolve to `/home/mechti/miniconda3/envs/deepgm-py312/bin/python`, use the full interpreter path explicitly for every Python command in this repo.
Install the core dependencies with `/home/mechti/miniconda3/envs/deepgm-py312/bin/python -m pip install -r requirements.txt`.
Run the current test suite with `/home/mechti/miniconda3/envs/deepgm-py312/bin/python -m unittest discover`.
Use separate training entry points for the two supervised tasks:
- `/home/mechti/miniconda3/envs/deepgm-py312/bin/python train_metal.py`
- `/home/mechti/miniconda3/envs/deepgm-py312/bin/python train_ec.py`
These wrappers enforce the requested training policy:
- separate runs for metal and EC, not one joint default run
- updated external features via `--external-feature-source updated`
- required ESM embeddings and required external features for loaded structures
- validation-enabled checkpoint selection by balanced accuracy
- batch size restricted to `8` or `16`
`train.py` still exists as the generic low-level entry point when you explicitly want to control the task yourself, including `joint`.
Training can now choose the external feature source with `--external-feature-source auto|bluues_rosetta|updated`.
Use `updated` to force `.data/updated_feature_extraction` and auto-generate missing JSON feature files for the training structures.
For Colab-friendly training defaults, use `python deepgm_colab.py` or `python -m deepgm_colab`.
There is also a ready-made notebook at `DeepGM_colab.ipynb` for running the same flow in Google Colab.
To build validated Colab dataset bundles from a local training tree, use `python build_colab_bundle.py`.
Use `.Study_Plan.md` for a 2.5-day guided ramp-up through the active code path.
The main runtime code lives in `training/`, `graph/`, `model.py`, and `train.py`.
The `prepare_training_and_test_set/` scripts are for dataset preparation, not the main training loop.
`requirements.txt` covers the core training and test stack used in the current repo.
If you work on embedding generation, you may need extra packages used by `embed_helpers/` in addition to the core requirements.
Modern replacements for the legacy Bluues/Rosetta external residue features live in `updated_feature_extraction/`.
They write structure-indexed JSON feature folders into `.data/updated_feature_extraction/`.

Colab example:

```bash
python deepgm_colab.py --mount-drive --epochs 3 --batch-size 4 --val-fraction 0.2
```

Dedicated local training examples:

```bash
/home/mechti/miniconda3/envs/deepgm-py312/bin/python train_metal.py --epochs 10 --batch-size 8
/home/mechti/miniconda3/envs/deepgm-py312/bin/python train_ec.py --epochs 10 --batch-size 16
```

Colab bundle example:

```bash
/home/mechti/miniconda3/envs/deepgm-py312/bin/python build_colab_bundle.py \
  --structure-dir /media/Data/pinmymetal_sets/mahomes/train_set \
  --summary-csv /media/Data/pinmymetal_sets/mahomes/train_set/data_summarizing_tables/final_data_summarazing_table_transition_metals_only_catalytic.csv \
  --esm-embeddings-dir /home/mechti/PycharmProjects/DeepGM/.data/embeddings \
  --output-dir /media/Data/pinmymetal_sets/mahomes/colab_bundle \
  --exclude-structure-id 1a16__chain_A__EC_3.4.11.9 \
  --exclude-structure-id 1aso__chain_A__EC_1.10.3.3
```

This writes:

- a manifest of included, unused, and invalid structures
- a `train_set_clean.tar.zst` archive with the train-set layout preserved
- an `embeddings_clean.tar.zst` archive with the matching ESM files

Default Colab paths used when you do not pass explicit paths:

- `structure_dir`: `/content/drive/MyDrive/DeepGM/train_set`
- `summary_csv`: `/content/drive/MyDrive/DeepGM/train_set/data_summarizing_tables/final_data_summarazing_table_transition_metals_only_catalytic.csv`
- `esm_embeddings_dir`: `/content/drive/MyDrive/DeepGM/embeddings`
- `runs_dir`: `/content/drive/MyDrive/DeepGM/training_runs`

Override them with normal training args such as `--structure-dir` and `--runs-dir`,
or by setting `DEEPGM_COLAB_STRUCTURE_DIR`, `DEEPGM_COLAB_EMBEDDINGS_DIR`,
`DEEPGM_COLAB_SUMMARY_CSV`, `DEEPGM_COLAB_RUNS_DIR`, and
`DEEPGM_COLAB_EXTERNAL_FEATURES_DIR`.

Updated feature generation example:

```bash
/home/mechti/miniconda3/envs/deepgm-py312/bin/python -m updated_feature_extraction.generate_features \
  --structure-dir /media/Data/pinmymetal_sets/mahomes/train_set \
  --output-root /home/mechti/PycharmProjects/DeepGM/.data/updated_feature_extraction \
  --skip-existing
```
