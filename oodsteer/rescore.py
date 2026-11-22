"""Regrade saved generations into a new directory; no GPU/model required."""
import argparse
import hashlib
import json
from pathlib import Path
from .answers import extract, SCORING_VERSION
from .report import summarize
from .data import COUNTS


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("input", help="A result JSONL or directory containing task-method JSONLs")
    p.add_argument("--output-dir")
    a=p.parse_args()
    source=Path(a.input).resolve()
    files=[source] if source.is_file() else sorted(f for f in source.glob("*.jsonl") if f.stem.split("-")[0] in COUNTS)
    if not files:
        p.error("No task result JSONLs found; pass the qwen3b directory")
    target=Path(a.output_dir) if a.output_dir else (source.parent if source.is_file() else source)/SCORING_VERSION
    if target.exists():
        raise FileExistsError(f"Choose a fresh output directory: {target}")
    target.mkdir(parents=True)
    summaries=[]
    for path in files:
        task=path.stem.split("-")[0]
        if task not in COUNTS:
            raise ValueError(f"Cannot infer task: {path}")
        rows=[json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        output=target/path.name
        with output.open("x",encoding="utf-8") as f:
            for r in rows:
                answer=extract(r["text"],task)
                updated=dict(r, previous_prediction=r["prediction"], previous_correct=r["correct"],
                    prediction=answer,correct=answer==r["gold"],scoring_version=SCORING_VERSION)
                f.write(json.dumps(updated,ensure_ascii=False)+"\n")
        meta_path=path.with_suffix(".meta.json")
        meta=json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        meta.update(scoring_version=SCORING_VERSION,source_file=str(path),
            source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            note="Rescoring only; generation, latency and truncation remain historical.")
        output.with_suffix(".meta.json").write_text(json.dumps(meta,indent=2),encoding="utf-8")
        summary=summarize(output)
        summary["complete"]=len(rows)==COUNTS[task]
        summaries.append(summary)
    (target/"summary.json").write_text(json.dumps(summaries,indent=2),encoding="utf-8")
    print(json.dumps(summaries,indent=2))
    print("Saved:",target)

if __name__=="__main__":
    main()
