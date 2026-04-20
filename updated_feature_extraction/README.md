# updated_feature_extraction

This package replaces the old Bluues and Rosetta-derived residue features with
a lighter workflow that is easier to install and maintain.

Chosen tools:

- `biotite` for structure parsing, solvent-accessibility calculation, residue
  geometry, backbone/side-chain dihedrals, and contact-based interaction
  proxies.
- `propka` for empirical pKa prediction and titration-environment terms.

The output keeps the same DeepGM feature names:

- `SASA`
- `BSA`
- `SolvEnergy`
- `fa_sol`
- `pKa_shift`
- `dpKa_desolv`
- `dpKa_bg`
- `dpKa_titr`
- `omega`
- `rama_prepro`
- `fa_dun`
- `fa_elec`
- `fa_atr`
- `fa_rep`

Important note:

These values are compatible replacements, not byte-for-byte recreations of
Rosetta/Bluues output. The package keeps the same feature contract for the
model, but the new terms are derived from Biotite geometry and PROPKA instead
of Rosetta score tables and Bluues electrostatics files.

Default output location:

- `.data/updated_feature_extraction/<structure_id>/residue_features.json`

Example:

```bash
/home/mechti/miniconda3/envs/deepgm-py312/bin/python -m updated_feature_extraction.generate_features \
  --structure-dir /media/Data/pinmymetal_sets/mahomes/train_set \
  --output-root /home/mechti/PycharmProjects/DeepGM/.data/updated_feature_extraction \
  --skip-existing
```

The training loader can read these JSON files directly. If
`.data/updated_feature_extraction` exists, DeepGM will prefer it automatically
when no explicit `--external-features-root-dir` is given.
