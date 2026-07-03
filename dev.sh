#!/usr/bin/env bash
# One command to run the whole thing: starts the API and the frontend together,
# and shuts both down cleanly on Ctrl-C.
#
#   ./dev.sh            (or, if it isn't executable:  bash dev.sh)
#
# The API runs on :8080, the visualizer on :5173 (open that one in your browser).
set -eo pipefail
cd "$(dirname "$0")"

API_PORT=8080
WEB_PORT=5173

# Free the ports first. The #1 reason "it won't start" is a leftover uvicorn or
# vite from an earlier run still holding the port, so we clear both up front.
for port in "$API_PORT" "$WEB_PORT"; do
  pids=$(lsof -ti "tcp:${port}" 2>/dev/null || true)
  if [ -n "$pids" ]; then
    echo "Freeing port ${port} (killing: ${pids})"
    echo "$pids" | xargs kill -9 2>/dev/null || true
  fi
done

# Activate the venv if present (so `uvicorn` resolves without manual sourcing).
if [ -d ".venv" ]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

if ! command -v uvicorn >/dev/null 2>&1; then
  echo "ERROR: uvicorn not found. Create the venv first:"
  echo "  python -m venv .venv && source .venv/bin/activate && pip install -e '.[dev]'"
  exit 1
fi
if ! command -v npm >/dev/null 2>&1; then
  echo "ERROR: npm not found. Install Node.js 18+ (https://nodejs.org)."
  exit 1
fi

# Install frontend deps on first run.
if [ ! -d "frontend/node_modules" ]; then
  echo "Installing frontend dependencies (first run)…"
  (cd frontend && npm install)
fi

echo ""
echo "  Precedent is starting…"
echo "  → API:        http://localhost:${API_PORT}"
echo "  → Visualizer: http://localhost:${WEB_PORT}   (open this one)"
echo ""

# Start the API in the background; make sure it dies when this script exits.
uvicorn precedent.api.main:app --port "$API_PORT" &
API_PID=$!
cleanup() { kill "$API_PID" 2>/dev/null || true; }
trap cleanup EXIT INT TERM

# Run the frontend in the foreground. When it exits (Ctrl-C), cleanup fires.
(cd frontend && npm run dev)
