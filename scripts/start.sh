#!/bin/sh
set -eu

PROJECT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$PROJECT_DIR"
exec ./.venv/bin/python -m uvicorn family_ai.app:app --host 0.0.0.0 --port 8000
