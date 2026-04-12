## Project Direction

1. Predict metal type from the coordination sphere plus nearby residue context.
2. Predict coarse reaction/mechanism class through the EC top-level label.

Current model direction:
- GVP over pocket graphs
- ESMC residue embeddings with late fusion
- Pocket/site geometry features such as `v_net`, `v_res`, and `cos(theta)`

---

## Things Codex Need To Fix

### Current Code State

Current main module split:
- `structure_parsing.py`: structure parsing and metal-pocket extraction
- `ring_edges.py`: RING path resolution and endpoint parsing
- `graph_construction.py`: graph assembly and pocket graph export
- `esm_feature_loading.py`: ESM embedding alignment and loading
- `training_feature_sources.py`: feature-source orchestration and coverage reporting
- `training_data.py`: supervised pocket loading and dataset orchestration

Cleanup status:
- The major oversized modules were split into smaller focused files.
- The code structure is now in a good enough state for functional work.
- Further refactoring is optional and should only happen if new features force it.

### 1. Wire the real feature pipeline into training

Status:
- Done at the code and validation level.

What was implemented:
- `training_data.py` now loads real ESM embeddings during pocket loading.
- `training_data.py` now loads real residue-level external features from the MAHOMES per-structure directories.
- The loader now distinguishes strict mode from fallback mode:
  - strict mode fails fast when required ESM or external features are missing
  - fallback mode is only used when explicitly allowed
- Feature coverage is now recorded in the training metadata:
  - residue and pocket coverage for ESM embeddings
  - residue and pocket coverage for external features
  - feature fallback records
- Residues now track whether their ESM and external features were actually attached.
- Auxiliary PDB files inside MAHOMES feature folders are excluded from the structure scan so structures are not double-counted.

Current behavior:
- Training no longer silently runs with zero-filled ESM/external features by default.
- `train.py` exposes:
  - `--esm-embeddings-dir`
  - `--external-features-root-dir`
  - `--allow-missing-esm-embeddings`
  - `--allow-missing-external-features`
- Run metadata now includes feature coverage and fallback reporting in `dataset_summary.json`.

Validation completed:
- Added focused tests in `tests/test_training_data_loading.py`.
- Verified ESM tensor-to-residue alignment on a known MAHOMES sample.
- Verified that the training loader attaches real ESM and external features.
- Verified that strict mode raises when required ESM embeddings are missing.
- The new tests passed in the local `.venv`.

Definition of done:
- `load_training_pockets_with_report_from_dir()` returns pockets with real `esm_embedding` and `external_features` data attached when those inputs exist.
- Strict mode now prevents accidental training on silently zero-filled feature inputs.

Remaining note:
- Step 2 is still needed to clean up the ESM artifact format itself. Step 1 currently supports the existing tensor-only files via alignment heuristics, which is sufficient for validation but not the final clean design.

---

### 2. Fix the ESM embedding artifact format

Status:
- Still open.

Problem:
- The embedding writer saves one tensor per chain.
- The training code expects a lookup keyed by residue identity `(chain_id, resseq, icode)`.
- These formats are not compatible.

Tasks:
- Redesign the embedding save format so each file contains:
  - structure id
  - chain id
  - residue order / residue ids
  - embedding tensor aligned to those residue ids
- Add a loader that converts the saved artifact into the lookup expected by `attach_esm_embeddings()`.
- Verify residue order alignment between BioPython parsing and the saved embedding artifact.
- Decide on one canonical location and naming scheme for embedding files and document it.

Definition of done:
- Embeddings written by `embed_helpers/esmc.py` can be loaded directly into the training pipeline without ad hoc conversion.
- Residue-id alignment is tested and deterministic.

Validation:
- Add a round-trip test: write sample embedding metadata, load it back, and assert the lookup keys and tensor lengths match the parsed residues.

---

### 3. Fix RING edge ingestion

Status:
- Still open.

Problem:
- Residue-metal `METAL_ION:SC_LIG` interactions from RING are currently discarded.
- RING edges are currently added in only one direction, while the radius graph is effectively bidirectional.

Tasks:
- Decide how metal-contact information should enter the graph:
  - as residue-residue edges with metal-contact flags, or
  - as explicit residue-metal relations/features
- Update `build_ring_interaction_edge_records()` so metal-contact information is not silently dropped.
- Symmetrize RING residue-residue edges during ingest unless there is a clear reason to keep them directional.
- Add graph-level reporting:
  - number of radius edges
  - number of RING residue-residue edges
  - number of residue-metal RING contacts recovered
  - number of duplicated / mirrored edges

Definition of done:
- Important RING chemistry is preserved in the graph representation.
- Edge direction policy is explicit and consistent.

Validation:
- Add a fixture using the sample `1a0e` RING file and assert:
  - `METAL_ION:SC_LIG` rows are accounted for
  - reverse edges exist when symmetry is expected

---

### 4. Fix and document the metal label scheme

Status:
- Partially done.

What is done:
- The active metal label space now matches the current catalytic summary:
  - `Zn`
  - `Cu`
  - merged `Co/Fe/Ni`
- The unused `Mn` class was removed from the active classifier label space.
- Added tests in `tests/test_label_schemes.py` to verify:
  - the active metal label mapping
  - the supported symbol-to-class mapping
  - that the current summary CSV has no `Mn` rows

Problem:
- The current summary file contains `CO`, `CU`, `FE`, `NI`, and `ZN`.
- The current classifier exposes a `Mn` class that is not present in the current training summary.
- Unsupported metal symbols currently raise hard errors.

Tasks:
- Decide the actual supervised metal taxonomy for the current dataset.
- Remove classes that are not present in the active training set, or switch to a configuration-driven label space.
- Decide how to handle unsupported metals:
  - skip with logging
  - map to `other`
  - fail only in strict mode
- Add a dataset label report before training starts.

Definition of done:
- The model head dimensions, label names, and dataset contents all agree.
- Loading does not crash unexpectedly when a structure contains a metal outside the active taxonomy.

Validation:
- Add a test that checks the label mapping against the current summary CSV distribution.
- Add a test for unsupported metal handling in strict and non-strict modes.

---

### 5. Add real test coverage for the data pipeline

Status:
- Partially done.

What is done:
- Added focused tests in `tests/test_training_data_loading.py`.
- Verified:
  - ESM alignment from saved tensor files to residue ids
  - loader attachment of ESM and external features
  - strict failure when required ESM embeddings are missing

What is still missing:
- broader tests for graph construction
- summary/site filtering tests
- metal clustering tests
- normalization tests
- a small model forward-pass smoke test

Problem:
- The repo has effectively no automated tests.
- `experiments/test_parsing_pdb.py` is empty.
- Current changes to graph construction or label parsing can regress silently.

Tasks:
- Replace the empty parsing test with real unit tests.
- Add coverage for:
  - structure id parsing
  - site filtering against the summary CSV
  - pocket extraction around clustered metals
  - node feature construction
  - graph edge construction
  - normalization stats shape/behavior
- Add one small end-to-end smoke test that builds a graph and runs one forward pass through the model.

Definition of done:
- Core data and graph logic is covered by repeatable tests.
- The repo has a standard test command that can run in one step.

Validation:
- `pytest` runs successfully in the intended environment and covers the core loader/graph/model path.

---

### 6. Add environment and dependency reproducibility

Status:
- Partially done locally, not done at the repo level.

What is done:
- A working local `.venv` was used to run the new validation tests.
- The local validation environment now includes `torch` CPU, `torch_geometric`, and `biopython`.

What is still missing:
- a committed dependency manifest
- setup instructions in the repo
- a standard one-command environment bootstrap

Problem:
- There is no dependency manifest in the repo.
- Local validation currently fails because the environment is missing `torch` and `pytest`.

Tasks:
- Add one canonical environment definition:
  - `pyproject.toml`, or
  - `requirements.txt`, or
  - `environment.yml`
- List the required runtime and dev dependencies explicitly.
- Document how to set up the environment and run tests/training.
- Make sure the chosen environment includes `torch`, `torch_geometric`, `biopython`, and `pytest`.

Definition of done:
- A fresh environment can be created without guessing package versions.
- Test and training commands are documented and reproducible.

Validation:
- Recreate the environment from scratch and run the test suite plus one smoke-test training command.

---

### 7. Improve training-time reporting and safety checks

Status:
- Partially done.

What is done:
- dataset summaries now include feature coverage
- training metadata now records feature fallback information
- strict loading now fails fast for required features

What is still missing:
- stronger label-distribution reporting
- split leakage checks
- explicit preflight checks for training viability
- better experiment metrics beyond raw accuracy

Problem:
- Several important dataset and experiment assumptions are implicit.
- The code has TODOs for metrics and reporting, but not enough checks for real experiments yet.

Tasks:
- Add dataset summary reporting for:
  - label distributions
  - feature coverage
  - pockets removed for missing labels/features
  - split leakage risk by selected grouping key
- Add a preflight check before training starts:
  - non-empty train/val
  - expected head label coverage
  - feature availability
  - edge coverage
- Decide and implement the primary evaluation metrics:
  - accuracy
  - balanced accuracy
  - macro-F1
  - per-class recall

Definition of done:
- Training failures happen early and clearly.
- Saved run artifacts are sufficient to understand what the model actually trained on.

Validation:
- Run one training job and confirm the saved metadata captures split, labels, feature coverage, and metrics.

---

## Later / Optional

- Evaluate whether supervised contrastive loss helps after the baseline is correct.
- Revisit alternative replacements or complements to BLUUES-derived features.
- Expand the dataset creation flow for predicted enzymatic/non-enzymatic sites only after the current supervised pipeline is reliable.

---

## Recommended Execution Order

1. Fix the ESM artifact format and make it the canonical loader path.
2. Fix RING edge ingestion and symmetry.
3. Align the metal label scheme with the real dataset.
4. Expand tests for graph construction, filtering, and smoke inference.
5. Add a committed environment/dependency file.
6. Improve training reporting and experiment checks.
