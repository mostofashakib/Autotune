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

# ── Frontend dependencies ────────────────────────────────────────────────────
FRONTEND_DIR="$ROOT/frontend"
if ! command -v npm &>/dev/null; then
  err "npm is required for the web UI. Install Node.js 20 or newer."
  exit 1
fi
if [[ ! -d "$FRONTEND_DIR/node_modules" ]]; then
  log "Installing web UI dependencies..."
  npm --prefix "$FRONTEND_DIR" install --silent --no-fund --no-audit
fi
ok "Web UI ready."

# ── Stop previous instances ──────────────────────────────────────────────────
"$ROOT/kill.sh" >/dev/null 2>&1 || true

log "Starting the API on ${CYAN}http://localhost:7860${RESET}"
uv run autotune api --host 127.0.0.1 --port 7860 &

log "Starting the web UI on ${CYAN}http://localhost:3001${RESET}"
echo -e "${BOLD}Press Ctrl+C to stop both.${RESET}"
echo ""
npm --prefix "$FRONTEND_DIR" run dev &

wait
