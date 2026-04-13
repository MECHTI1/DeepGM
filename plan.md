## Project Direction

1. Predict metal type from the coordination sphere plus nearby residue context.
2. Predict coarse reaction/mechanism class through the EC top-level label.

Current model direction:
- GVP over pocket graphs
- ESM residue embeddings with graph features
- Pocket/site geometry features such as `v_net`, `v_res`, and angle-derived terms

---

## Current Code State

The active runtime path is now organized around:
- `train.py`
- `training/config.py`
- `training/run.py`
- `training/data.py`
- `training/structure_loading.py`
- `training/feature_sources.py`
- `training/esm_feature_loading.py`
- `training/site_filter.py`
- `training/labels.py`
- `training/graph_dataset.py`
- `training/loop.py`
- `training/preflight.py`
- `training/splits.py`
- `graph/construction.py`
- `graph/structure_parsing.py`
- `graph/edge_building.py`
- `graph/feature_utils.py`
- `graph/ring_edges.py`
- `model.py`

Cleanup status:
- The oversized training/data logic was split into focused modules.
- The training loop, split logic, and graph-dataset logic are now separated cleanly.
- The current structure is good enough for feature work and experiments.
- Further refactoring should be driven by concrete needs, not style alone.

---

## Completed Recently

### 1. Real feature loading is wired into training

Status:
- Done.

What is in place:
- Training loads real ESM embeddings during pocket loading.
- Training loads real residue-level external features from the feature directories.
- Strict mode fails fast for missing required ESM or external features.
- Dataset summaries include feature coverage and feature fallback reporting.
- Auxiliary structure files are excluded from structure scanning to avoid double counting.

Validation:
- Covered by `tests/test_training_data_loading.py`.

---

### 2. Core training orchestration is cleaner

Status:
- Done.

What changed:
- Training orchestration lives in `training/run.py`.
- Split logic lives in `training/splits.py`.
- Epoch/train/eval helpers live in `training/loop.py`.
- Early run validation and graphability checks now live in `training/preflight.py`.
- Graph dataset and normalization logic live in `training/graph_dataset.py`.
- A regression around recorded `train_loss` was fixed and tested.

Validation:
- Covered by `tests/test_training_run.py`.

---

### 3. Active metal label space is aligned with the current dataset

Status:
- Done at the current policy level.

What is in place:
- Active classes are now:
  - `Zn`
  - `Cu`
  - merged `Co/Fe/Ni`
- The active classifier label space no longer includes the old unused `Mn` class.
- Label mapping tests exist.

Current policy:
- Unsupported metals are now handled explicitly through runtime policy:
  - `error`
  - `skip`

Remaining note:
- An `other` class is still a later design choice, not a current requirement.

Validation:
- Covered by `tests/test_label_schemes.py`.

---

### 4. Basic automated tests now exist

Status:
- Done at the baseline level.

Current state:
- The repo has a real `unittest` test suite under `tests/`.
- Current passing baseline is 32 tests.
- Coverage exists for:
  - label mapping
  - site filtering helpers
  - training data loading
  - graph helper behavior
  - split/reporting logic in training
  - one end-to-end graph-to-model smoke path
  - graph construction at the runtime boundary
  - normalization behavior
  - clustered-metal extraction
  - RING ingestion behavior

Current standard command:
- `./.venv/bin/python -m unittest discover -s tests`

---

### 5. Repo-level dependency bootstrap exists

Status:
- First pass done.

What is in place:
- `requirements.txt` now defines the core runtime/test dependencies.
- `README.md` now gives a short setup and test path.

Remaining gap:
- This still needs one clean environment recreation check from scratch.

---

## What Still Needs Work And Why

The project no longer needs broad cleanup for elegance.

What it does need now is narrower and more practical:
- protect the data/model contracts that can break silently
- make environment setup reproducible for someone new to the repo
- remove format ambiguity in the embedding artifacts
- tighten experiment safety before longer training runs

That means the remaining work is mostly about correctness, reproducibility, and clearer failure modes, not another structural rewrite.

---

## Open Work

### 1. End-to-end smoke test for graph -> model forward

Status:
- Done.

What is in place:
- A smoke test now builds a real `PocketGraphDataset`.
- It batches graphs with the PyG `DataLoader`.
- It runs `GVPPocketClassifier.forward()`.
- It asserts `logits_metal`, `logits_ec`, `embed`, and `loss`.

Why it was needed:
- This is the most fragile runtime boundary after a refactor.
- It protects tensor names, tensor shapes, batching behavior, and supervised-loss wiring.

Validation:
- Covered by `tests/test_model_smoke.py`.

---

### 2. Fix the ESM embedding artifact format

Status:
- Done in code.

What is now in place:
- The canonical saved payload includes explicit `residue_ids`.
- The loader can read the canonical payload directly.
- The embedding writer now emits the canonical payload format.
- Round-trip coverage exists for canonical write/load alignment.

What still remains:
- Regenerate or migrate the full embedding artifact set for the real dataset.
- Confirm the full-dataset artifact inventory is complete before baseline training.

Definition of done:
- The normal path uses the canonical residue-aligned payload.
- Residue-id alignment is deterministic and tested.

---

Why this still matters:
- The code path is now correct.
- The remaining risk is operational: whether the full artifact set has actually been produced.

---

### 3. Fix RING edge ingestion and symmetry policy

Status:
- Done in a first explicit policy pass.

What is now in place:
- Residue-residue RING edges are ingested with an explicit bidirectional policy.
- `METAL_ION:SC_LIG` rows are preserved instead of being silently dropped.
- The current metal-contact policy is explicit in code.
- Runtime-boundary tests cover the new behavior.

Current policy:
- Residue-residue RING edges are mirrored.
- Residue-metal contacts are represented as residue self-loop RING signals.

Definition of done:
- Important RING chemistry is preserved.
- Edge direction behavior is explicit and validated.

Why this still matters:
- This is about graph semantics, not code style.
- If edge meaning is wrong or inconsistent, model quality and interpretation both suffer.

Remaining note:
- The main remaining question is experimental, not structural:
  - whether the current residue self-loop encoding is the best representation after baseline runs

---

### 4. Add training preflight checks and stronger reporting

Status:
- Done in a first strong pass.

What is already present:
- Dataset summaries already include feature coverage and label distributions.
- A first-pass preflight report now exists and is attached to the dataset summary.
- Early checks now cover empty splits, empty-residue pockets, and graph construction across all train and validation pockets.
- Leakage checks now exist for the selected split key.
- Label-viability checks now fail early.
- Feature-viability coverage is included in the preflight report.
- Stronger saved metrics now include:
  - balanced accuracy
  - macro-F1
  - per-class recall
- Checkpoint selection metric is now explicit and configurable.

What is still missing:
- real baseline evidence that the chosen selection metric and loss policy are the right defaults

Remaining task:
- Confirm the chosen model-selection metric against the first real baseline.

Definition of done:
- Training fails early and clearly on bad inputs.
- Saved run artifacts are enough to interpret the experiment.

Why this still matters:
- The code now fails earlier than before, which is good.
- But experiments can still fail late or produce hard-to-interpret artifacts if preflight remains too shallow.

Current note:
- The remaining uncertainty is now mostly about experiment choice, not missing safety code.

---

### 5. Expand test coverage around graph construction

Status:
- Done at the current must-have level.

What is now covered:
- pocket extraction around clustered metals
- graph construction at the `pocket_to_pyg_data()` level
- normalization stats behavior
- RING ingestion behavior

Definition of done:
- Core graph-building logic is covered by repeatable tests, not just helper-level tests.

Why this still matters:
- Current tests are good enough for baseline confidence.
- They are not yet enough to fully protect the structure-parsing and graph-building path.

---

### 6. Move from `requirements.txt` to a fuller packaging story later

Status:
- Optional later.

What is already good enough now:
- `requirements.txt` is sufficient for bootstrapping the current repo.
- `README.md` gives a short setup and test path.

What may be cleaner later:
- a `pyproject.toml`
- separated dev/runtime dependency groups
- one reproducible environment recreation check

Why this is later, not urgent:
- The current repo needed an explicit dependency manifest more than it needed a perfect packaging design.
- `requirements.txt` solves the immediate reproducibility gap well enough.

---

## Recommended Execution Order

1. Regenerate or verify the full canonical ESM artifact set.
2. Run one baseline training job on the catalytic-only set.
3. Hand-check summary-to-pocket mapping on a small sample.
4. Confirm the model-selection metric and loss policy from baseline evidence.
5. Decide later whether to replace `requirements.txt` with `pyproject.toml`.

---

## Current Validation Snapshot

Date:
- 2026-04-13

What was checked now:
- The current `unittest` suite passes:
  - 32 tests
- Strict catalytic loading was probed against the real mounted dataset.
- Summary-to-pocket matching was hand-checked on a small real sample.
- `deepgm-py312` was confirmed to have the ESM runtime needed for canonical embedding generation.
- Canonical ESM artifacts were generated successfully for sampled real catalytic structures:
  - `1a0e__chain_A__EC_5.3.1.5`
  - `1afr__chain_A__EC_1.14.99.6`

What is confirmed:
- The codebase is stable enough to keep iterating without another broad manual code-review pass first.
- The catalytic summary-site matching is behaving correctly on sampled structures.
- External residue-feature directories are present for sampled real structures.
- Canonical ESM generation works from the current machine in `deepgm-py312`.
- Newly generated canonical artifacts are picked up by the strict training loader.

Current blocker:
- The remaining blocker is not generation logic, but incomplete inventory in the runtime embeddings directory.
- Current snapshot:
  - runtime embeddings in `.data/embeddings`: 3 files
  - sampled real catalytic structures generated successfully: 2
- The full catalytic set still does not have the canonical inventory needed for baseline training.

What this means:
- The next real blocker is operational data availability, not missing training-path code.
- A full baseline run should wait for the canonical ESM inventory to exist in the runtime embeddings location.

Practical conclusion:
- Do not spend time manually rereading the whole codebase before continuing.
- Use the built-in checks first, then only inspect code manually if one of those checks fails.

---

## Must Do Now

### A. Must Implement In Code Now

These were the code changes that had to land before longer training runs or new feature work.
They are now done at the current required level:

1. Canonicalize the ESM embedding artifact format.
   - Done in code.

2. Fix RING metal-contact ingestion and make edge symmetry policy explicit.
   - Done in code.

3. Strengthen training preflight and experiment reporting.
   - Done in code.

4. Make unsupported metal handling an explicit policy.
   - Done in code for `error` and `skip`.

5. Expand graph-construction tests at the real runtime boundaries.
   - Done at the current must-have level.

Remaining code question:
- Done.
- The current final baseline policy is:
  - class-balanced cross-entropy on both heads
  - equal task weights for metal and EC losses
- This is now explicit in `model.py`.

### B. Must Validate Experimentally Now

These are not architecture tasks, but they are still required now for safe iteration:

1. Verify or regenerate the full canonical ESM artifact inventory.
   - This is the active blocker right now.
   - The code path is ready and generation was validated from `deepgm-py312`.
   - What remains is to run the batch generation across the real dataset until the runtime embeddings inventory is complete.
   - Current generation entry point:
     - `embed_helpers/esmc.py`

2. Run one baseline training job on the catalytic-only training set.
   - Record the first reference metrics.
   - This should happen immediately after item 1, not before.

3. Hand-check summary-to-pocket mapping on a small sample.
   - Verify that the catalytic-only summary rows match the pockets that are actually retained.
   - A first manual sample check was done and looked correct.

4. Record retained dataset counts before training.
   - Number of retained catalytic pockets.
   - Class distributions after filtering and splitting.
   - Best done by the runtime loaders/preflight path once the ESM inventory is in place.

5. Confirm that the chosen validation metric is the right one for model selection.
   - The code now supports explicit selection.
   - The remaining question is whether the default should stay as-is after the first baseline.

6. Validate the now-explicit loss policy on the first real baseline.
   - Current policy:
     - class-balanced CE for metal
     - class-balanced CE for EC
     - equal task weights
   - Revisit only if the first real baseline shows a clear failure mode.

### C. Can Wait

These are useful, but they are not the current blockers:

1. Replace `requirements.txt` with a fuller packaging story.
   - `pyproject.toml`
   - dev/runtime dependency groups
   - cleaner environment management

2. Additional cleanup not driven by correctness, reproducibility, or experiment safety.
   - The codebase is already structured well enough for active work.

3. Loss redesign beyond the first baseline-supported decision.
   - Do not optimize the loss policy further before baseline evidence exists.

---

## Bottom Line

The project is not in a "harsh" state.

The code is now structured well enough for real work, and the main must-do-now code items are in place.

The best next steps are now operational and experimental rather than architectural:
- finish generating the canonical ESM artifacts into the runtime embeddings location
- rerun the retained-dataset/preflight path on the real catalytic set
- run and record the first baseline
- use that baseline to confirm checkpoint selection and validate the chosen loss policy
