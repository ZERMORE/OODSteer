import json
import sys
from pathlib import Path
import numpy as np


def summarize(path):
    rows = [json.loads(s) for s in Path(path).read_text(encoding="utf-8").splitlines() if s.strip()]
    if not rows:
        raise ValueError("Empty results")
    x = np.array([r["correct"] for r in rows],dtype=float)
    rng = np.random.default_rng(42)
    samples = np.array([rng.choice(x,len(x),replace=True).mean() for _ in range(10000)])
    # Cold first example excluded from latency summary, but never from accuracy.
    warm = rows[1:] or rows
    result = dict(path=str(path),n=len(rows),accuracy_percent=100*x.mean(),ci95_percent=(100*np.quantile(samples,[.025,.975])).tolist(),parse_failures=sum(r["prediction"] is None for r in rows),truncated=sum(r["truncated"] for r in rows),ms_per_token=1000*sum(r["seconds"] for r in warm)/max(1,sum(r["tokens"] for r in warm)))
    return result

if __name__ == "__main__":
    for path in sys.argv[1:]:
        print(json.dumps(summarize(path),ensure_ascii=False))
