import json
import re
from decimal import Decimal, InvalidOperation
from pathlib import Path

COUNTS = {"csqa":1221, "strategyqa":229, "gsm8k":1319, "aqua":254}
MARKERS = {"csqa":"So the answer is", "strategyqa":"So the answer is", "gsm8k":"The answer is", "aqua":"The answer is"}


def extract(text, task):
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


def load(task, local=None, allow_partial=False):
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
        from datasets import load_dataset
        if task == "csqa":
            ds = load_dataset("tau/commonsense_qa", split="validation", cache_dir=str(paths["datasets"]))
            rows = [dict(id=r["id"], question=r["question"], choices=[f"({k}) {v}" for k,v in zip(r["choices"]["label"], r["choices"]["text"])], answer=r["answerKey"]) for r in ds]
        elif task == "gsm8k":
            ds = load_dataset("openai/gsm8k", "main", split="test", cache_dir=str(paths["datasets"]))
            rows = [dict(id=str(i), question=r["question"], answer=str(Decimal(r["answer"].split("####")[-1].strip().replace(",", "")).normalize())) for i,r in enumerate(ds)]
        elif task == "aqua":
            ds = load_dataset("deepmind/aqua_rat", "raw", split="validation", cache_dir=str(paths["datasets"]))
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
