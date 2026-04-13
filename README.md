# DeepGM

Create a virtual environment and install the core dependencies with `python -m venv .venv` and `./.venv/bin/pip install -r requirements.txt`.
Run the current test suite with `./.venv/bin/python -m unittest discover -s tests`.
The current training entry point is `./.venv/bin/python train.py`.
Use `.Study_Plan.md` for a 2.5-day guided ramp-up through the active code path.
The main runtime code lives in `training/`, `graph/`, `model.py`, and `train.py`.
The `prepare_training_and_test_set/` scripts are for dataset preparation, not the main training loop.
`requirements.txt` covers the core training and test stack used in the current repo.
If you work on embedding generation, you may need extra packages used by `embed_helpers/` in addition to the core requirements.
