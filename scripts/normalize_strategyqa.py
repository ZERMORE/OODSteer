"""Convert user-supplied original dev JSON (list or qid->record) to local schema."""
import argparse
import json
from pathlib import Path
p=argparse.ArgumentParser()
p.add_argument("input")
p.add_argument("output", nargs="?", help="Default: <data-root>/datasets/strategyqa/strategyqa-dev.jsonl")
p.add_argument("--data-root")
a=p.parse_args()
raw=json.loads(Path(a.input).read_text(encoding="utf-8"))
if isinstance(raw,dict):
    rows=[dict(v,id=k) for k,v in raw.items()]
else:
    rows=raw
if len(rows)!=229:
    raise ValueError(f"Expected original 229-example dev, got {len(rows)}; no automatic random split")
out=[]
for i,r in enumerate(rows):
    answer=r['answer']
    answer='yes' if answer is True else 'no' if answer is False else str(answer).lower()
    if answer not in ('yes','no'):
        raise ValueError(answer)
    out.append(dict(id=str(r.get('qid',r.get('id',i))),question=r['question'],answer=answer))
from oodsteer.paths import storage_paths
target=Path(a.output) if a.output else storage_paths(a.data_root)["strategyqa"]
target.parent.mkdir(parents=True,exist_ok=True)
with target.open('x',encoding='utf-8') as f:
    for r in out:
        f.write(json.dumps(r,ensure_ascii=False)+'\n')
