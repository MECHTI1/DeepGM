# DeepGM

Use the project interpreter `/home/mechti/miniconda3/envs/deepgm-py312/bin/python`.
Before running project code, verify:
- `which python`
- `python -c "import sys; print(sys.executable)"`
If either command does not resolve to `/home/mechti/miniconda3/envs/deepgm-py312/bin/python`, use the full interpreter path explicitly for every Python command in this repo.
Install the core dependencies with `/home/mechti/miniconda3/envs/deepgm-py312/bin/python -m pip install -r requirements.txt`.
Run the current test suite with `/home/mechti/miniconda3/envs/deepgm-py312/bin/python -m unittest discover`.
The current training entry point is `/home/mechti/miniconda3/envs/deepgm-py312/bin/python train.py`.
Use `.Study_Plan.md` for a 2.5-day guided ramp-up through the active code path.
The main runtime code lives in `training/`, `graph/`, `model.py`, and `train.py`.
The `prepare_training_and_test_set/` scripts are for dataset preparation, not the main training loop.
`requirements.txt` covers the core training and test stack used in the current repo.
If you work on embedding generation, you may need extra packages used by `embed_helpers/` in addition to the core requirements.
