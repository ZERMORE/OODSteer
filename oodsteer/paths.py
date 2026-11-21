"""Shared server storage layout; configure before importing HF libraries."""
import os
from pathlib import Path

DEFAULT_DATA_ROOT = "/home/wxh/wdl/data"


def data_root(value=None):
    return Path(value or os.environ.get("OODSTEER_DATA_ROOT") or DEFAULT_DATA_ROOT).expanduser()


def storage_paths(root=None):
    base = data_root(root)
    return {
        "root": base,
        "hf_home": base / "huggingface",
        "models": base / "models" / "huggingface",
        "datasets": base / "datasets" / "huggingface",
        "strategyqa": base / "datasets" / "strategyqa" / "strategyqa-dev.jsonl",
        "assets": base / "cache" / "huggingface-assets",
        "results": base / "results" / "OODSteer",
        "logs": base / "logs" / "OODSteer",
    }


def configure_storage(root=None):
    paths = storage_paths(root)
    for key in ("hf_home", "models", "datasets", "assets", "results", "logs"):
        paths[key].mkdir(parents=True, exist_ok=True)
    # Explicit policy: the project root option overrides generic HF env vars.
    os.environ.update({
        "OODSTEER_DATA_ROOT": str(paths["root"]),
        "HF_HOME": str(paths["hf_home"]),
        "HF_HUB_CACHE": str(paths["models"]),
        "HUGGINGFACE_HUB_CACHE": str(paths["models"]),
        "TRANSFORMERS_CACHE": str(paths["models"]),
        "HF_DATASETS_CACHE": str(paths["datasets"]),
        "HF_ASSETS_CACHE": str(paths["assets"]),
    })
    return paths
