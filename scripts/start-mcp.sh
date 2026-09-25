#!/bin/sh
set -eu

PROJECT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR/src"
exec "$PROJECT_DIR/.venv/bin/python" -m family_ai.mcp_server
