# TODO

- Fix pocket-level label assignment for multi-pocket structures in the training loader.
- Current behavior in `train_utils.py` skips labeled structures when multiple extracted pockets exist without a stable mapping key.
- This can silently shrink and bias the supervised dataset, so it should be resolved before trusting real training results.
