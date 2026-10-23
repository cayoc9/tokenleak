#!/usr/bin/env bash
# TokenLeak -- Experiment runner for remote GPU server
# =====================================================
# SECURITY NOTE: This file is excluded from git via .gitignore
# Do NOT commit or share this file. It may reference server-specific
# environment variables. Use only generic "remote GPU server" language
# in any published documentation.
# =====================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="${PYTHON:-python3}"

echo "[*] Installing dependencies..."
"$PYTHON" -m pip install -r "$REPO_ROOT/requirements.txt" --quiet

echo "[*] Running TokenLeak simulation evaluation..."
cd "$REPO_ROOT"
"$PYTHON" -m src.evaluation.evaluate

echo "[+] Experiment complete. Results in data/results/"
