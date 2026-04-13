# DeepGM Plan

Last updated: 2026-04-13

This file is the short project plan for the current codebase state. It is not a
historical dump. It should answer three questions quickly:

1. What is the project trying to do?
2. What is already in good shape?
3. What still matters next?

## Project Direction

Current supervised tasks:

1. Predict metal class from the local coordination environment and nearby
   residue context.
2. Predict coarse enzyme class through the EC top-level label.

Current model direction:

- GVP-style pocket graph encoder
- ESM residue embeddings fused with graph features
- pocket/site geometry features such as `v_net`, `v_res`, and angle-derived
  terms

## Current Code State

The active runtime path is organized around:

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

The current structure is good enough for feature work, model experiments, and
bug fixing. The project no longer needs a broad rewrite.

## What Is In Good Shape

### Runtime structure

- Training orchestration is centralized in `training/run.py`.
- Split logic, loop logic, preflight checks, and dataset normalization are split
  into focused modules.
- The main training path is readable in pipeline order.

### Data and feature loading

- Real ESM embeddings and residue-level external features are wired into the
  runtime loaders.
- Summary-based catalytic-site filtering is explicit.
- Missing required features fail clearly in strict mode.
- Feature coverage is reported in the dataset summary.

### Graph construction

- Graph creation is tested at the `pocket_to_pyg_data()` boundary.
- Shell-role calculation in the graph path is now pure and does not mutate the
  input `ResidueRecord` objects.
- Radius-edge construction now avoids a blind all-pairs scan by using a
  broad-phase spatial filter before exact atom-level checks.
- RING residue-residue symmetry and residue-metal handling are explicit.

### Training behavior

- Preflight checks catch empty splits, leakage, empty-residue pockets, and graph
  construction failures early.
- Checkpoint selection is explicit.
- The default selection behavior is coherent:
  - `train_loss` when there is no validation split
  - validation metrics only when validation is enabled
- Class-balanced cross-entropy is explicit for both supervised heads.

### Tests

- The repo has a real `unittest` suite under `tests/`.
- Current passing baseline: 40 tests.
- Coverage exists for label logic, site filtering, training data loading, graph
  helpers, runtime graph construction, normalization, training orchestration,
  and graph-to-model smoke behavior.

Standard test command:

```bash
./.venv/bin/python -m unittest discover -s tests
```

## Current Priorities

These are the next tasks that matter most.

### 1. Verify real-data readiness

The main remaining operational risk is not training-path code. It is whether the
real runtime artifact inventory is complete enough for baseline runs.

Need to confirm:

- required ESM artifacts exist for the intended dataset
- required external feature directories exist for the intended dataset
- strict catalytic loading succeeds on a meaningful sample, then on the real
  training set

Why this matters:

- If the runtime inventory is incomplete, baseline training will fail for
  operational reasons rather than modeling reasons.

### 2. Run and record one baseline training job

Once the real-data inventory is confirmed, run one baseline and save:

- retained pocket counts
- class distributions after filtering and splitting
- selected checkpoint metric
- train and validation metrics from the saved run artifacts

Why this matters:

- The current code is stable enough that the next useful evidence should come
  from one real baseline, not more structural cleanup.

### 3. Validate current default training policies with evidence

The code now supports explicit policy choices, but they still need baseline
evidence behind them.

Validate:

- whether the chosen selection metric is the right default
- whether equal task weighting is acceptable
- whether the current merged metal-label policy is sufficient for the dataset in
  practice

Why this matters:

- These are now experiment questions, not architecture questions.

## Lower-Priority Work

These are worthwhile, but not current blockers.

### Packaging and environment polish

- move from `requirements.txt` to a fuller packaging story later
- separate runtime and dev dependencies if the repo needs cleaner onboarding
- run one clean environment recreation check from scratch

### Further performance work

- if pocket sizes grow, revisit graph-building performance again
- if training throughput becomes a bottleneck, profile data loading and graph
  creation before changing model code

### Additional tests

- add more real-data integration checks only when they protect an actual failure
  mode
- avoid growing the suite with tests that duplicate helper-level coverage

## What Not To Do Next

- Do not start another broad refactor without a concrete runtime problem.
- Do not redesign the loss policy before at least one real baseline exists.
- Do not optimize legacy prep scripts unless they are blocking the current
  training workflow.

## Immediate Next Step

Use strict loading on the intended real dataset, confirm the artifact inventory,
then run one baseline training job and use that run to drive the next decision.
