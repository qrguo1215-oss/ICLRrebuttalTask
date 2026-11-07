#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
CMD=run
if [[ "${1:-}" =~ ^(run|smoke|plan|verify)$ ]]; then CMD="$1"; shift; fi
BASE="${PYTHON_BIN:-python3}"
INSTALL=1
ARGS=()
while (( $# )); do
  case "$1" in
    --python) BASE="${2:?missing python}"; shift 2 ;;
    --skip-install) INSTALL=0; shift ;;
    *) ARGS+=("$1"); shift ;;
  esac
done
export PYTHONNOUSERSITE=1 TOKENIZERS_PARALLELISM=false
if [[ "$CMD" == plan || "$CMD" == verify ]]; then
  exec "$BASE" task.py "$CMD" "${ARGS[@]}"
fi
"$BASE" -c 'import sys; assert (3,10) <= sys.version_info[:2] <= (3,11), "Use Python 3.10 or 3.11"'
if (( INSTALL )); then
  if [[ ! -x .venv/bin/python ]]; then "$BASE" -m venv .venv; fi
  PY="$ROOT/.venv/bin/python"
  EXPECTED="$(sha256sum requirements.txt | cut -d' ' -f1)"
  if [[ ! -f .venv/.requirements.sha256 ]] || [[ "$(<.venv/.requirements.sha256)" != "$EXPECTED" ]]; then
    "$PY" -m pip install --upgrade pip
    "$PY" -m pip install -r requirements.txt
    printf '%s\n' "$EXPECTED" > .venv/.requirements.sha256
  fi
else
  PY="$BASE"
fi
"$PY" -m unittest discover -s tests -v
exec "$PY" task.py "$CMD" "${ARGS[@]}"
