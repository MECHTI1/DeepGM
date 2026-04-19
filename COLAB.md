# DeepGM Colab Run

This file is the shortest path to running the current cleaned DeepGM bundle in
Google Colab.

The code stays normal Python code. Colab is only another environment to run it
in.

## What You Need In Drive

Put these files somewhere in Google Drive first:

- `train_set_clean.tar.zst`
- `embeddings_clean.tar.zst`

Current bundle output on the local machine:

- `/media/Data/pinmymetal_sets/mahomes/colab_bundle/train_set_clean.tar.zst`
- `/media/Data/pinmymetal_sets/mahomes/colab_bundle/embeddings_clean.tar.zst`

## Colab Cells

### 1. Mount Drive

```python
from google.colab import drive
drive.mount("/content/drive")
```

### 2. Clone The Repo

```bash
%cd /content
!git clone <YOUR_GITHUB_REPO_URL> DeepGM
%cd /content/DeepGM
```

### 3. Install Core Dependencies

```bash
!python -m pip install -r requirements.txt
```

### 4. Copy Or Read Bundles From Drive

Edit the Drive paths below to where you uploaded the bundle files.

```bash
!mkdir -p /content/DeepGM_data
!cp "/content/drive/MyDrive/DeepGM/train_set_clean.tar.zst" /content/DeepGM_data/
!cp "/content/drive/MyDrive/DeepGM/embeddings_clean.tar.zst" /content/DeepGM_data/
```

If you want to avoid the copy step, you can unpack directly from Drive, but
training is usually better when the data itself lives under `/content`.

### 5. Unpack The Bundles

```bash
!mkdir -p /content/DeepGM_data
!tar --zstd -xf /content/DeepGM_data/train_set_clean.tar.zst -C /content/DeepGM_data
!tar --zstd -xf /content/DeepGM_data/embeddings_clean.tar.zst -C /content/DeepGM_data
```

After this, the expected paths should exist:

- `/content/DeepGM_data/train_set`
- `/content/DeepGM_data/embeddings`

### 6. Run Training

This uses the Colab wrapper but still goes through the normal training code.
If you unpacked to the default Drive layout under `/content/drive/MyDrive/DeepGM`,
the wrapper now defaults `--structure-dir`, `--summary-csv`, `--esm-embeddings-dir`,
and `--runs-dir` automatically. The explicit command below remains valid and is
still the clearest way to run it.

```bash
!python deepgm_colab.py \
  --structure-dir /content/DeepGM_data/train_set \
  --summary-csv /content/DeepGM_data/train_set/data_summarizing_tables/final_data_summarazing_table_transition_metals_only_catalytic.csv \
  --esm-embeddings-dir /content/DeepGM_data/embeddings \
  --runs-dir /content/drive/MyDrive/DeepGM/training_runs \
  --device cuda \
  --epochs 10 \
  --batch-size 4 \
  --val-fraction 0.2
```

## Optional Quick Sanity Run

Before a longer job:

```bash
!python deepgm_colab.py \
  --structure-dir /content/DeepGM_data/train_set \
  --summary-csv /content/DeepGM_data/train_set/data_summarizing_tables/final_data_summarazing_table_transition_metals_only_catalytic.csv \
  --esm-embeddings-dir /content/DeepGM_data/embeddings \
  --runs-dir /content/drive/MyDrive/DeepGM/training_runs \
  --device cuda \
  --epochs 1 \
  --batch-size 2 \
  --val-fraction 0.1
```

## Notes

- The cleaned bundle was built from the manifest in
  `/media/Data/pinmymetal_sets/mahomes/colab_bundle/colab_bundle_manifest.json`.
- The bundle excludes known problematic structures that were blocking strict
  loading.
- `deepgm_colab.py` is only a wrapper. The main training path is still
  `train.py` and `training/run.py`.
- The standard Colab flow assumes the bundle already contains the ESM embedding
  files. If you want the runtime to generate missing embeddings on the fly, you
  also need to install the `esm` package used by `embed_helpers/esmc.py`.
- Automatic RING edge generation is not part of the default Colab path. That
  helper still expects an external `ring` executable, so keep using prebuilt
  ring-edge files unless you plan to install that binary yourself.
