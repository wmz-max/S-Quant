#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if [[ -n "${SQUANT_PYTHON:-}" ]]; then
    interpreter="$SQUANT_PYTHON"
elif [[ -x .venv/bin/python ]]; then
    interpreter="$PWD/.venv/bin/python"
else
    interpreter=python
fi
exec "$interpreter" -m squant "$@"
