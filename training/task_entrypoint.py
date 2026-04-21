from __future__ import annotations

from typing import Sequence

from training.config import TrainConfig, default_selection_metric_for_task, parse_args

DEFAULT_SEPARATE_TASK_VAL_FRACTION = 0.2
DEFAULT_SEPARATE_TASK_BATCH_SIZE = 8
DEFAULT_SEPARATE_EXTERNAL_FEATURE_SOURCE = "updated"
VALID_SEPARATE_TASK_BATCH_SIZES = (8, 16)


def cli_option_present(argv: Sequence[str], option: str) -> bool:
    option_prefix = f"{option}="
    return any(arg == option or arg.startswith(option_prefix) for arg in argv)


def apply_separate_task_defaults(task: str, argv: Sequence[str] | None = None) -> list[str]:
    effective_argv = list(argv or [])
    updated_argv = list(effective_argv)

    if not cli_option_present(effective_argv, "--task"):
        updated_argv.extend(["--task", task])
    if not cli_option_present(effective_argv, "--val-fraction"):
        updated_argv.extend(["--val-fraction", str(DEFAULT_SEPARATE_TASK_VAL_FRACTION)])
    if not cli_option_present(effective_argv, "--batch-size"):
        updated_argv.extend(["--batch-size", str(DEFAULT_SEPARATE_TASK_BATCH_SIZE)])
    if not cli_option_present(effective_argv, "--external-feature-source"):
        updated_argv.extend(["--external-feature-source", DEFAULT_SEPARATE_EXTERNAL_FEATURE_SOURCE])

    return updated_argv


def validate_separate_task_config(config: TrainConfig, *, expected_task: str) -> None:
    if config.task != expected_task:
        raise ValueError(
            f"Dedicated {expected_task} training only supports --task {expected_task!r}, got {config.task!r}."
        )
    if config.external_feature_source != DEFAULT_SEPARATE_EXTERNAL_FEATURE_SOURCE:
        raise ValueError(
            "Dedicated task training requires --external-feature-source updated "
            "so the updated residue feature set is always used."
        )
    if not config.require_esm_embeddings:
        raise ValueError("Dedicated task training requires ESM embeddings for every loaded structure.")
    if not config.require_external_features:
        raise ValueError("Dedicated task training requires updated external features for every loaded structure.")
    if config.val_fraction <= 0.0:
        raise ValueError("Dedicated task training requires --val-fraction > 0 to select checkpoints by balanced accuracy.")
    expected_metric = default_selection_metric_for_task(expected_task, has_validation=True)
    if config.selection_metric != expected_metric:
        raise ValueError(
            f"Dedicated {expected_task} training requires --selection-metric {expected_metric!r}, "
            f"got {config.selection_metric!r}."
        )
    if config.batch_size not in VALID_SEPARATE_TASK_BATCH_SIZES:
        raise ValueError(
            "Dedicated task training requires --batch-size to be one of "
            f"{', '.join(str(value) for value in VALID_SEPARATE_TASK_BATCH_SIZES)}."
        )


def parse_separate_task_args(expected_task: str, argv: Sequence[str] | None = None) -> TrainConfig:
    config = parse_args(apply_separate_task_defaults(expected_task, argv))
    validate_separate_task_config(config, expected_task=expected_task)
    return config
