#!/bin/sh
set -eu

RUNTIME_DIR="* Support/FamilyAI/app"
cd "$RUNTIME_DIR"
export PYTHONPATH="$RUNTIME_DIR/src"
exec "$RUNTIME_DIR/.venv/bin/python" -m uvicorn family_ai.app:app --host 0.0.0.0 --port 8000
