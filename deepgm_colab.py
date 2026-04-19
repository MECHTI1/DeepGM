from __future__ import annotations

import argparse
import os
import sys
from dataclasses import replace
from pathlib import Path
from typing import Sequence

import torch

from training.config import TrainConfig, build_arg_parser as build_training_arg_parser, parse_args
from training.run import run_training

COLAB_DRIVE_ROOT_ENV = "DEEPGM_COLAB_DRIVE_ROOT"
COLAB_STRUCTURE_DIR_ENV = "DEEPGM_COLAB_STRUCTURE_DIR"
COLAB_EMBEDDINGS_DIR_ENV = "DEEPGM_COLAB_EMBEDDINGS_DIR"
COLAB_RUNS_DIR_ENV = "DEEPGM_COLAB_RUNS_DIR"
COLAB_EXTERNAL_FEATURES_DIR_ENV = "DEEPGM_COLAB_EXTERNAL_FEATURES_DIR"
DEFAULT_COLAB_DRIVE_ROOT = Path("/content/drive/MyDrive")
DEFAULT_COLAB_PROJECT_DIRNAME = "DeepGM"
DEFAULT_COLAB_MOUNT_ROOT = Path("/content/drive")


def build_wrapper_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--mount-drive",
        action="store_true",
        help="Mount Google Drive when running inside Google Colab.",
    )
    parser.add_argument(
        "--drive-root",
        type=Path,
        default=Path(os.getenv(COLAB_DRIVE_ROOT_ENV, DEFAULT_COLAB_DRIVE_ROOT)),
        help=(
            "Base Google Drive directory used for Colab-friendly defaults. "
            f"Default: ${COLAB_DRIVE_ROOT_ENV} or {DEFAULT_COLAB_DRIVE_ROOT}"
        ),
    )
    return parser


def build_colab_arg_parser() -> argparse.ArgumentParser:
    parser = build_training_arg_parser()
    parser.description = (
        "Train DeepGM with Colab-friendly defaults. "
        "Pass standard training args as usual; this wrapper only supplies safer Colab defaults."
    )
    group = parser.add_argument_group("colab")
    group.add_argument(
        "--mount-drive",
        action="store_true",
        help="Mount Google Drive when running inside Google Colab.",
    )
    group.add_argument(
        "--drive-root",
        type=Path,
        default=Path(os.getenv(COLAB_DRIVE_ROOT_ENV, DEFAULT_COLAB_DRIVE_ROOT)),
        help=(
            "Base Google Drive directory used for Colab-friendly defaults. "
            f"Default: ${COLAB_DRIVE_ROOT_ENV} or {DEFAULT_COLAB_DRIVE_ROOT}"
        ),
    )
    return parser


def cli_option_present(argv: Sequence[str], option: str) -> bool:
    option_prefix = f"{option}="
    return any(arg == option or arg.startswith(option_prefix) for arg in argv)


def default_colab_structure_dir(drive_root: Path) -> Path:
    configured = os.getenv(COLAB_STRUCTURE_DIR_ENV)
    if configured:
        return Path(configured).expanduser()
    return drive_root / DEFAULT_COLAB_PROJECT_DIRNAME / "train_set"


def default_colab_embeddings_dir(drive_root: Path) -> str:
    configured = os.getenv(COLAB_EMBEDDINGS_DIR_ENV)
    if configured:
        return str(Path(configured).expanduser())
    return str(drive_root / DEFAULT_COLAB_PROJECT_DIRNAME / "embeddings")


def default_colab_runs_dir(drive_root: Path) -> str:
    configured = os.getenv(COLAB_RUNS_DIR_ENV)
    if configured:
        return str(Path(configured).expanduser())
    return str(drive_root / DEFAULT_COLAB_PROJECT_DIRNAME / "training_runs")


def default_colab_external_features_dir() -> str | None:
    configured = os.getenv(COLAB_EXTERNAL_FEATURES_DIR_ENV)
    if configured:
        return str(Path(configured).expanduser())
    return None


def apply_colab_defaults(
    config: TrainConfig,
    *,
    argv: Sequence[str],
    drive_root: Path,
) -> TrainConfig:
    updates: dict[str, object] = {}

    if not cli_option_present(argv, "--structure-dir"):
        updates["structure_dir"] = default_colab_structure_dir(drive_root)
    if not cli_option_present(argv, "--esm-embeddings-dir"):
        updates["esm_embeddings_dir"] = default_colab_embeddings_dir(drive_root)
    if not cli_option_present(argv, "--runs-dir"):
        updates["runs_dir"] = default_colab_runs_dir(drive_root)
    if not cli_option_present(argv, "--external-features-root-dir"):
        default_external_dir = default_colab_external_features_dir()
        if default_external_dir is not None:
            updates["external_features_root_dir"] = default_external_dir
    if not cli_option_present(argv, "--device"):
        updates["device"] = "cuda" if torch.cuda.is_available() else "cpu"

    if not updates:
        return config
    return replace(config, **updates)


def maybe_mount_google_drive(should_mount: bool) -> None:
    if not should_mount:
        return
    try:
        from google.colab import drive
    except ImportError as exc:
        raise RuntimeError(
            "--mount-drive was requested, but google.colab is not available in this Python environment."
        ) from exc
    drive.mount(str(DEFAULT_COLAB_MOUNT_ROOT))


def validate_colab_runtime_inputs(config: TrainConfig) -> None:
    if not Path(config.structure_dir).exists():
        raise FileNotFoundError(
            "Structure directory does not exist: "
            f"{config.structure_dir}. In Colab, mount Drive and pass --structure-dir "
            f"or set {COLAB_STRUCTURE_DIR_ENV}."
        )

    if config.esm_embeddings_dir is not None:
        Path(config.esm_embeddings_dir).mkdir(parents=True, exist_ok=True)
    if config.runs_dir is not None:
        Path(config.runs_dir).mkdir(parents=True, exist_ok=True)


def parse_colab_args(
    argv: Sequence[str] | None = None,
) -> tuple[argparse.Namespace, TrainConfig, list[str]]:
    effective_argv = list(argv) if argv is not None else sys.argv[1:]

    if "-h" in effective_argv or "--help" in effective_argv:
        build_colab_arg_parser().parse_args(effective_argv)
        raise AssertionError("argparse help should exit before reaching this point")

    wrapper_args, training_argv = build_wrapper_arg_parser().parse_known_args(effective_argv)
    config = parse_args(training_argv)
    config = apply_colab_defaults(
        config,
        argv=effective_argv,
        drive_root=wrapper_args.drive_root.expanduser(),
    )
    return wrapper_args, config, training_argv


def main(argv: Sequence[str] | None = None) -> None:
    wrapper_args, config, _training_argv = parse_colab_args(argv)
    maybe_mount_google_drive(wrapper_args.mount_drive)
    validate_colab_runtime_inputs(config)

    print(f"DeepGM Colab structure dir: {config.structure_dir}")
    print(f"DeepGM Colab embeddings dir: {config.esm_embeddings_dir}")
    print(f"DeepGM Colab runs dir: {config.runs_dir}")
    print(f"DeepGM Colab device: {config.device}")

    run_dir = run_training(config)
    print(f"DeepGM Colab run completed: {run_dir}")


if __name__ == "__main__":
    main()
