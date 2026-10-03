#!/bin/zsh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
ENGINE="$ROOT/engine"

export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

if [[ ! -f "$ROOT/agent_state.json" || ! -d "$ENGINE" ]]; then
  echo "Brawl Draft Local has not been set up yet."
  echo "Run setup.command first."
  read "?Press Return to close this window."
  exit 1
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is not on PATH. Run setup.command again."
  read "?Press Return to close this window."
  exit 1
fi

cd "$ENGINE"

# Refresh the released-brawler catalogue at most once per day. This happens
# before the local server starts, never during a live draft. Network failure is
# non-fatal: update_roster.py keeps the last good cache/model roster.
uv run python "$ROOT/update_roster.py" --max-age-hours 24 || true
uv run python "$ROOT/app.py"
