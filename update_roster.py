from __future__ import annotations

import argparse
import json
import os
import tempfile
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "brawler_roster.json"
DEFAULT_URL = "https://api.brawlapi.com/v1/brawlers"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Refresh the current Brawl Stars brawler roster.")
    p.add_argument("--force", action="store_true", help="refresh even if the cache is fresh")
    p.add_argument(
        "--max-age-hours",
        type=float,
        default=24.0,
        help="skip network refresh when the cache is newer than this many hours",
    )
    p.add_argument("--source-file", type=Path, help="read API-shaped JSON from a local file (tests/offline)")
    p.add_argument("--quiet", action="store_true")
    return p.parse_args()


def cache_age_hours(path: Path) -> float | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        stamp = payload.get("updated_at")
        if not stamp:
            return None
        dt = datetime.fromisoformat(stamp.replace("Z", "+00:00"))
        return (datetime.now(timezone.utc) - dt.astimezone(timezone.utc)).total_seconds() / 3600.0
    except Exception:
        return None


def read_source(args: argparse.Namespace) -> tuple[dict, str]:
    if args.source_file:
        return json.loads(args.source_file.read_text(encoding="utf-8")), str(args.source_file)

    url = os.environ.get("BRAWL_ROSTER_URL", DEFAULT_URL)
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "BrawlDraftLocal/1.2 (+local roster refresh)"},
    )
    with urllib.request.urlopen(req, timeout=6) as response:
        return json.load(response), url


def write_atomic(payload: dict) -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=".brawler_roster_", suffix=".json", dir=OUT.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
            f.write("\n")
        os.replace(temp_name, OUT)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def main() -> int:
    args = parse_args()
    age = cache_age_hours(OUT)
    if not args.force and age is not None and age <= args.max_age_hours:
        if not args.quiet:
            cached = json.loads(OUT.read_text(encoding="utf-8"))
            print(f"Roster cache is fresh: {cached.get('count', '?')} brawlers ({age:.1f}h old).")
        return 0

    try:
        raw, source = read_source(args)
        items = raw.get("list")
        if not isinstance(items, list):
            raise ValueError("roster endpoint did not return a list")

        names = sorted(
            {
                str(item.get("name", "")).strip().upper()
                for item in items
                if isinstance(item, dict)
                and item.get("released", True) is not False
                and str(item.get("name", "")).strip()
            }
        )
        if len(names) < 50:
            raise ValueError(f"roster response looks incomplete ({len(names)} names)")

        payload = {
            "source": source,
            "updated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "count": len(names),
            "brawlers": names,
        }
        write_atomic(payload)
        if not args.quiet:
            print(f"Updated roster: {len(names)} released brawlers → {OUT.name}")
        return 0
    except Exception as exc:
        # Roster freshness should never prevent the local draft app from opening.
        # app.py will keep using the last good cache, or fall back to model vocab.
        if not args.quiet:
            if OUT.exists():
                print(f"Roster refresh skipped ({exc}). Keeping the existing cache.")
            else:
                print(f"Roster refresh unavailable ({exc}). The app will use the model roster.")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
