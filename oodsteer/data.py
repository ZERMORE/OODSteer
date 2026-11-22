import json
import re
from decimal import Decimal, InvalidOperation
from pathlib import Path

COUNTS = {"csqa":1221, "strategyqa":229, "gsm8k":1319, "aqua":254}
MARKERS = {"csqa":"So the answer is", "strategyqa":"So the answer is", "gsm8k":"The answer is", "aqua":"The answer is"}


def extract_legacy(text, task):
    marker = MARKERS[task]
    matches = list(re.finditer(re.escape(marker), text, re.I))
    if not matches:
        return None
    tail = text[matches[-1].end():].strip()
    if task in ("csqa", "aqua"):
        m = re.match(r"[(:\s]*([a-e])\b", tail, re.I)
        return m[1].upper() if m else None
    if task == "strategyqa":
        m = re.match(r"[(:\s]*(yes|no)\b", tail, re.I)
        return m[1].lower() if m else None
    m = re.match(r"[(:\s$]*([-+]?\d[\d,]*(?:\.\d+)?)", tail)
    if not m:
        return None
    try:
        return str(Decimal(m[1].replace(",", "")).normalize())
    except InvalidOperation:
        return None


def load(task, local=None, allow_partial=False, recipe=None, metadata=None):
    from .paths import storage_paths
    paths = storage_paths()
    if task == "strategyqa" and local is None:
        local = paths["strategyqa"]
    if local:
        if not Path(local).is_file():
            raise FileNotFoundError(f"Dataset file missing: {local}; provide the original dev set or use --data")
        rows = [json.loads(s) for s in Path(local).read_text(encoding="utf-8").splitlines() if s.strip()]
        for r in rows:
            if not all(k in r for k in ("id", "question", "answer")):
                raise ValueError("Local JSONL requires id, question, answer; optional choices")
    else:
        from datasets import load_dataset, DownloadConfig
        from .paths import default_recipe, dataset_cache_candidates
        specs = {"csqa": ("tau/commonsense_qa", None, "validation"),
                 "gsm8k": ("openai/gsm8k", "main", "test"),
                 "aqua": ("deepmind/aqua_rat", "raw", "validation")}
        repo_id, config, split = specs[task]
        recipe_path = Path(recipe) if recipe else default_recipe()
        revision = None
        if recipe_path is not None:
            record = json.loads(recipe_path.read_text(encoding="utf-8"))["datasets"][task]
            if (record["repo_id"], record["config"], record["split"]) != specs[task]:
                raise ValueError(f"Incompatible recipe dataset: {recipe_path} / {task}")
            revision = record["revision"]
            if not revision:
                raise ValueError("Recipe must pin a dataset revision")
        ds = None
        used_cache = None
        # Probe each old cache without allowing downloads. If none works, the
        # regular load writes into OODmethod's canonical shared cache.
        for cache in dataset_cache_candidates():
            if not cache.is_dir():
                continue
            try:
                ds = load_dataset(repo_id, name=config, split=split, revision=revision,
                                  cache_dir=str(cache),
                                  download_config=DownloadConfig(local_files_only=True))
                used_cache = str(cache)
                break
            except (OSError, ValueError, ConnectionError) as exc:
                print(f"Cache unavailable: {cache}: {type(exc).__name__}", flush=True)
        if ds is None:
            used_cache = str(paths["datasets"])
            ds = load_dataset(repo_id, name=config, split=split, revision=revision,
                              cache_dir=used_cache)
        if metadata is not None:
            metadata.update(repo_id=repo_id, config=config, split=split,
                            revision=revision, cache_dir=used_cache,
                            recipe=str(recipe_path) if recipe_path else None,
                            fingerprint=getattr(ds, "_fingerprint", None))
        if task == "csqa":
            rows = [dict(id=r["id"], question=r["question"], choices=[f"({k}) {v}" for k,v in zip(r["choices"]["label"], r["choices"]["text"])], answer=r["answerKey"]) for r in ds]
        elif task == "gsm8k":
            rows = [dict(id=str(i), question=r["question"], answer=str(Decimal(r["answer"].split("####")[-1].strip().replace(",", "")).normalize())) for i,r in enumerate(ds)]
        elif task == "aqua":
            rows = [dict(id=str(i), question=r["question"], choices=r["options"], answer=r["correct"]) for i,r in enumerate(ds)]
        else:
            raise ValueError("StrategyQA: provide original 229-example dev set via --data; never substitute a random split")
    if len({str(r["id"]) for r in rows}) != len(rows):
        raise ValueError("Duplicate sample IDs")
    if not allow_partial and len(rows) != COUNTS[task]:
        raise ValueError(f"Expected {COUNTS[task]} examples, got {len(rows)}")
    for r in rows:
        if task in ("csqa", "aqua"):
            r["answer"] = str(r["answer"]).upper()
        elif task == "gsm8k":
            r["answer"] = str(Decimal(str(r["answer"]).replace(",", "")).normalize())
        elif task == "strategyqa":
            a = r["answer"]
            r["answer"] = "yes" if a is True else "no" if a is False else str(a).lower()
            if r["answer"] not in ("yes", "no"):
                raise ValueError("StrategyQA gold answer must be yes/no or bool")
    return rows


def prompt(row, task, prefix):
    question = row["question"]
    if row.get("choices"):
        question += " " + " ".join(row["choices"])
    return prefix.rstrip() + "\n\nQ: " + question + "\nA:"


# Public scorer used by inference and regrading. Legacy protocol stays available.
from .answers import extract
