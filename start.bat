#!/usr/bin/env bash
set -euo pipefail

export FLASK_ENV=development
export FLASK_APP=app.py

python3 app.py
