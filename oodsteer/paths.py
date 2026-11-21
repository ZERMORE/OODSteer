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
        "hf_home": base / "cache" / "huggingface",
        "models": base / "cache" / "huggingface" / "hub",
        "datasets": base / "cache" / "huggingface" / "datasets",
        "strategyqa": base / "datasets" / "strategyqa" / "strategyqa-dev.jsonl",
        "assets": base / "cache" / "huggingface" / "assets",
        "results": base / "results" / "OODSteer",
        "logs": base / "logs" / "OODSteer",
        "tmp": base / "tmp",
        "xet": base / "cache/huggingface/xet",
    }


def configure_storage(root=None):
    paths = storage_paths(root)
    for key in ("hf_home", "models", "datasets", "assets", "results", "logs", "tmp", "xet"):
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
        "HF_XET_CACHE": str(paths["xet"]),
        "TMPDIR": str(paths["tmp"]),
    })
    return paths


def dataset_cache_candidates(root=None):
    base = data_root(root)
    return [base / "cache/huggingface/datasets", base / "huggingface/datasets",
            base / "datasets/huggingface"]


def default_recipe(root=None):
    base = data_root(root) / "results/OODmethod"
    # Prefer the protocol-matched run; never choose an arbitrary experiment.
    for name in ("qwen25-paper-v4-matched", "qwen25-paper-v4"):
        path = base / name / "recipe.json"
        if path.is_file():
            return path
    return None


def resolve_model(model, root=None):
    """Reuse only the exact HF repo, never another size or a VL checkpoint."""
    if Path(model).is_dir() or "/" not in model:
        return model
    base = data_root(root)
    from huggingface_hub import try_to_load_from_cache
    for hub in (base / "cache/huggingface/hub", base / "huggingface/hub",
                base / "models/huggingface"):
        config = try_to_load_from_cache(model, "config.json", cache_dir=str(hub))
        if isinstance(config, str):
            folder = Path(config).parent
            if (folder / "model.safetensors").is_file():
                return str(folder)
            index = folder / "model.safetensors.index.json"
            if index.is_file():
                import json
                shards = set(json.loads(index.read_text())["weight_map"].values())
                if all((folder / shard).is_file() for shard in shards):
                    return str(folder)
    return model
