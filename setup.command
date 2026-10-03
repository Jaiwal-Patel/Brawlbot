#!/bin/zsh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
ENGINE="$ROOT/engine"
TMP="$ROOT/.setup_tmp"

echo ""
echo "══════════════════════════════════════════════"
echo "  Brawl Draft Local — first-time setup"
echo "══════════════════════════════════════════════"
echo ""

command -v curl >/dev/null 2>&1 || {
  echo "curl is required but was not found."
  exit 1
}

if ! command -v uv >/dev/null 2>&1; then
  echo "Installing uv (Python environment manager)..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
fi

if ! command -v uv >/dev/null 2>&1; then
  echo ""
  echo "uv was installed but is not yet on PATH."
  echo "Close Terminal, reopen it, then run setup.command again."
  exit 1
fi

echo "Installing Python 3.13 if needed..."
uv python install 3.13

rm -rf "$TMP"
mkdir -p "$TMP"

echo "Downloading the current draft-engine source..."
curl -fL \
  "https://github.com/elixf7/brawlstars-draft-agent/archive/refs/heads/main.zip" \
  -o "$TMP/engine.zip"

unzip -q "$TMP/engine.zip" -d "$TMP"
rm -rf "$ENGINE"
mv "$TMP/brawlstars-draft-agent-main" "$ENGINE"
rm -rf "$TMP"

cd "$ENGINE"

echo "Installing engine dependencies..."
uv sync

CONFIG="$(
  for f in configs/season*.toml; do
    [[ -e "$f" ]] || continue
    base="${f:t:r}"
    number="${base#season}"
    printf '%s\t%s\n' "$number" "$f"
  done | sort -n | tail -1 | cut -f2-
)"
if [[ -z "$CONFIG" ]]; then
  echo "No season config was found in the engine repository."
  exit 1
fi

echo ""
echo "Using latest available config: $CONFIG"
echo "Training the compact local FFM model..."
uv run bsdraft-train fm -c "$CONFIG"

echo ""
echo "Building map, matchup, counter, and synergy lookup data..."
uv run python "$ROOT/bootstrap.py" "$CONFIG" "$ROOT/agent_state.json"

echo ""
echo "Refreshing the current released-brawler roster..."
uv run python "$ROOT/update_roster.py" --force || true

echo ""
echo "Running a quick engine smoke test..."
uv run python "$ROOT/smoke_test.py"

echo ""
echo "══════════════════════════════════════════════"
echo "  Setup complete."
echo "  Double-click run.command to launch the app."
echo "══════════════════════════════════════════════"
echo ""
read "?Press Return to close this window."
