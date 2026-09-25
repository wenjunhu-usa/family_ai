#!/bin/sh
set -eu

RUNTIME_DIR="* Support/FamilyAI/app"
cd "$RUNTIME_DIR"
export PYTHONPATH="$RUNTIME_DIR/src"
export OLLAMA_HOST="http://127.0.0.1:11434"
exec "$RUNTIME_DIR/.venv/bin/python" -m family_ai.curator scheduled
