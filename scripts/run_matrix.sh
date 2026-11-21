#!/usr/bin/env bash
set -euo pipefail
# Default checkout: /home/wxh/projects/OODSteer
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
PRESET="${1:-qwen3b}"
MODEL="${2:-Qwen/Qwen2.5-3B-Instruct}"
RUN="${3:-main}"
DATA_ROOT="${OODSTEER_DATA_ROOT:-/home/wxh/wdl/data}"
RESULT_DIR="$DATA_ROOT/results/OODSteer/$RUN/$PRESET"
LOG_DIR="$DATA_ROOT/logs/OODSteer/$RUN/$PRESET"
mkdir -p "$RESULT_DIR" "$LOG_DIR"
for task in csqa strategyqa gsm8k aqua; do
  for method in standard usteer; do
    result="$RESULT_DIR/$task-$method.jsonl"
    log="$LOG_DIR/$task-$method.log"
    if [[ -e "$result" || -e "$log" ]]; then
      echo "Output already exists; choose a fresh RUN: $result / $log" >&2
      exit 1
    fi
    python -u -m oodsteer.cli --data-root "$DATA_ROOT" --model "$MODEL" --preset "$PRESET" --task "$task" --method "$method" --prompt-file "prompts/$task.txt" --output "$result" 2>&1 | tee "$log"
  done
done
python -m oodsteer.report "$RESULT_DIR"/*.jsonl
