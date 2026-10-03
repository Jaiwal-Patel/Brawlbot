#!/bin/zsh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
ENGINE="$ROOT/engine"
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

if [[ ! -d "$ENGINE" ]]; then
  echo "Brawl Draft Local has not been set up yet. Run setup.command first."
  read "?Press Return to close this window."
  exit 1
fi

cd "$ENGINE"
uv run python "$ROOT/update_roster.py" --force

echo ""
echo "Roster refresh complete. Restart run.command if the app is already open."
read "?Press Return to close this window."
