#!/usr/bin/env bash
set -euo pipefail
# Run from project root. Model may also be a local model directory.
PRESET="${1:-qwen3b}"
MODEL="${2:-Qwen/Qwen2.5-3B-Instruct}"
RUN="${3:-main}"
for task in csqa strategyqa gsm8k aqua; do
  extra=()
  if [[ "$task" == strategyqa ]]; then
    extra+=(--data data/strategyqa-dev.jsonl)
  fi
  for method in standard usteer; do
    python -m oodsteer.cli --model "$MODEL" --preset "$PRESET" --task "$task"       --method "$method" --prompt-file "prompts/$task.txt" "${extra[@]}"       --output "results/$RUN/$PRESET/$task-$method.jsonl"
  done
done
python -m oodsteer.report results/"$RUN"/"$PRESET"/*.jsonl
