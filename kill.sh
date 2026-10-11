#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Autotune — Process Killer
# Gracefully stops running Autotune services and workers. Safe to run anytime.
# ─────────────────────────────────────────────────────────────────────────────

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'
BOLD='\033[1m'; RESET='\033[0m'

ok()   { echo -e "  ${GREEN}✓${RESET} $*"; }
warn() { echo -e "  ${YELLOW}⚠${RESET} $*"; }
info() { echo -e "  ${BOLD}·${RESET} $*"; }

echo ""
echo -e "${BOLD}Stopping Autotune processes...${RESET}"
echo ""

killed_any=false

kill_port() {
  local port="$1"
  local label="$2"
  local pids
  pids=$(lsof -ti:"$port" 2>/dev/null || true)
  if [[ -n "$pids" ]]; then
    echo "$pids" | xargs kill -TERM 2>/dev/null || true
    sleep 0.5
    local remaining
    remaining=$(lsof -ti:"$port" 2>/dev/null || true)
    if [[ -n "$remaining" ]]; then
      echo "$remaining" | xargs kill -9 2>/dev/null || true
    fi
    ok "$label (port $port) stopped — PIDs: $pids"
    killed_any=true
  else
    info "Nothing running on port $port ($label)"
  fi
}

kill_port 8080 "Autotune API"
kill_port 8501 "Autotune Dashboard"

kill_pattern() {
  local pattern="$1"
  local label="$2"
  local pids
  pids=$(pgrep -f "$pattern" 2>/dev/null || true)
  if [[ -n "$pids" ]]; then
    echo "$pids" | xargs kill -TERM 2>/dev/null || true
    sleep 0.3
    local remaining
    remaining=$(pgrep -f "$pattern" 2>/dev/null || true)
    if [[ -n "$remaining" ]]; then
      echo "$remaining" | xargs kill -9 2>/dev/null || true
    fi
    ok "$label stopped — PIDs: $pids"
    killed_any=true
  fi
}

kill_pattern "autotune.presentation" "Autotune worker/CLI"

echo ""
if [[ "$killed_any" == true ]]; then
  echo -e "${GREEN}All Autotune processes stopped.${RESET}"
else
  echo -e "${BOLD}No running Autotune processes found.${RESET}"
fi
echo ""
