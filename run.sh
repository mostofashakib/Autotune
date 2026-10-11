#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Autotune — Local Runner
# Checks dependencies with uv, clears existing ports, and launches Autotune.
# Ctrl+C gracefully stops all processes.
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$ROOT/.venv"

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; RESET='\033[0m'

log()  { echo -e "${BOLD}[autotune]${RESET} $*"; }
ok()   { echo -e "${GREEN}  ✓${RESET} $*"; }
warn() { echo -e "${YELLOW}  ⚠${RESET} $*"; }
err()  { echo -e "${RED}  ✗${RESET} $*"; }

cleanup() {
  echo ""
  log "Shutting down..."
  "$ROOT/kill.sh" >/dev/null 2>&1 || true
  log "All processes stopped cleanly."
  exit 0
}

trap cleanup INT TERM

echo ""
echo -e "${BOLD}======================================================${RESET}"
echo -e "${BOLD} Autotune — Autonomous AI Training & Benchmarking     ${RESET}"
echo -e "${BOLD}======================================================${RESET}"
echo ""

# ── Dependency Check ─────────────────────────────────────────────────────────
if ! command -v uv &>/dev/null; then
  err "uv is required. Install it via: curl -LsSf https://astral.sh/uv/install.sh | sh"
  exit 1
fi

if [[ ! -d "$VENV_DIR" ]]; then
  log "Creating virtual environment with uv..."
  uv venv "$VENV_DIR"
fi

log "Verifying package installation..."
uv pip install -e "$ROOT" --quiet

ok "Environment ready."

# ── Stop previous instances ──────────────────────────────────────────────────
"$ROOT/kill.sh" >/dev/null 2>&1 || true

log "Autotune CLI is ready. Showing available commands:"
echo ""
uv run python -m autotune.presentation.cli.main --help

echo ""
ok "Autotune is active. Run commands using: uv run autotune [inspect|train|evaluate]"
echo -e "${BOLD}Press Ctrl+C to exit.${RESET}"
echo ""

# Keep alive for interactive sessions
while true; do
  sleep 1
done
