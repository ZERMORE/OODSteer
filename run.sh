#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
ACTION="${1:-help}"
if [[ "$#" -gt 0 ]]; then shift; fi
case "$ACTION" in
  help|-h|--help)
    echo 'Usage: bash run.sh {setup|check|smoke|main|rescore} [options]'
    echo 'setup: create/update conda environment oodsteer and install dependencies'
    echo 'check: verify environment, GPU and tests'
    echo 'smoke: check and run 3 GSM8K examples with Standard + USteer'
    echo 'main: check and run complete datasets with Standard + USteer'
    echo 'Options for check/smoke/main: --gpu 0 --tasks gsm8k csqa aqua --run-name NAME'
    echo 'Default model: Qwen/Qwen2.5-3B-Instruct. All data: /home/wxh/wdl/data'
    exit 0 ;;
  setup|check|smoke|main|rescore) ;;
  *) echo "Unknown action: $ACTION" >&2; exit 2 ;;
esac
source "$ROOT/scripts/server_env.sh"
if [[ -n "${CONDA_EXE:-}" && -x "$CONDA_EXE" ]]; then
  CONDA_BIN="$CONDA_EXE"
elif command -v conda >/dev/null 2>&1; then
  CONDA_BIN="$(command -v conda)"
else
  echo 'Conda not found. Open a terminal with conda available, then retry.' >&2
  exit 1
fi
if [[ "$ACTION" == setup ]]; then
  if [[ "$#" -gt 0 ]]; then echo 'setup accepts no options' >&2; exit 2; fi
  if ! "$CONDA_BIN" run -n oodsteer python -c 'import sys' >/dev/null 2>&1; then
    "$CONDA_BIN" create -n oodsteer python=3.10 -y
  fi
  "$CONDA_BIN" run --no-capture-output -n oodsteer python -c 'import sys; assert sys.version_info[:2] == (3,10), "Existing oodsteer environment must use Python 3.10"'
  "$CONDA_BIN" run --no-capture-output -n oodsteer python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
  "$CONDA_BIN" run --no-capture-output -n oodsteer python -m pip install -e "$ROOT"
  "$CONDA_BIN" run --no-capture-output -n oodsteer python -m pip check
  echo 'Setup complete. Next: bash run.sh smoke'
else
  if [[ "$ACTION" == rescore ]]; then
    exec "$CONDA_BIN" run --no-capture-output -n oodsteer python -m oodsteer.rescore "$@"
  fi
  exec "$CONDA_BIN" run --no-capture-output -n oodsteer python "$ROOT/scripts/server.py" "$ACTION" "$@"
fi
