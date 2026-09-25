#!/bin/sh
set -eu

cd *
exec ./.venv/bin/python -m uvicorn family_ai.app:app --host 0.0.0.0 --port 8000
