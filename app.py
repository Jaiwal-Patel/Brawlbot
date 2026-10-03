from __future__ import annotations

import hashlib
import json
import math
import mimetypes
import pickle
import threading
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

import numpy as np

from bsdraft.data.matchup_db import MatchupDB, skill_ns_to_tier
from bsdraft.fm.ffm import FFMInference
from bsdraft.mcts.ffm_evaluator import FFMEvaluator
from bsdraft.mcts.evaluator import FMEvaluator
from bsdraft.mcts.recommend import recommend

from roster_support import augment_ffm_with_roster, load_roster_catalog


ROOT = Path(__file__).resolve().parent
STATIC = ROOT / "static"
STATE_PATH = ROOT / "agent_state.json"
PORT = 8765
HOST = "127.0.0.1"

if not STATE_PATH.exists():
    raise SystemExit("agent_state.json is missing. Run setup.command first.")

STATE = json.loads(STATE_PATH.read_text(encoding="utf-8"))

# The selectable Ranked map pool is intentionally separate from the training
# dataset. A new/featured map can therefore appear in the UI immediately even
# when the currently trained model has not seen enough matches from it yet.
CATALOG_PATH = ROOT / "ranked_maps.json"
if CATALOG_PATH.exists():
    MAP_CATALOG = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
else:
    MAP_CATALOG = {}

OBSERVED_MAP_MODES: dict[str, list[str]] = STATE.get(
    "observed_map_modes", STATE.get("map_modes", {})
)
MAP_MODES: dict[str, list[str]] = MAP_CATALOG.get("map_modes") or OBSERVED_MAP_MODES

with open(STATE["model_path"], "rb") as f:
    TRAINED = pickle.load(f)

if hasattr(TRAINED, "maps"):
    NATIVE_MODEL_MAPS = set(getattr(TRAINED, "maps", ()))
else:
    _schema = getattr(TRAINED, "schema", None)
    NATIVE_MODEL_MAPS = set(getattr(_schema, "maps", ())) if _schema is not None else set()
APPROXIMATE_MAPS: set[str] = set()

if hasattr(TRAINED, "vocab"):
    NATIVE_MODEL_BRAWLERS = set(getattr(TRAINED, "vocab", ()))
else:
    _schema = getattr(TRAINED, "schema", None)
    NATIVE_MODEL_BRAWLERS = set(getattr(_schema, "vocab", ())) if _schema is not None else set()

ROSTER_PATH = ROOT / "brawler_roster.json"
ROSTER_VOCAB, ROSTER_META = load_roster_catalog(ROSTER_PATH, NATIVE_MODEL_BRAWLERS)
APPROXIMATE_BRAWLERS: set[str] = set()

def _augment_ffm_with_ranked_maps(model: FFMInference) -> set[str]:
    """Add catalog-only maps using a neutral mode-average map embedding.

    The upstream FFM normally maps an unknown map name to embedding index 0.
    That is an arbitrary map and can distort a recommendation. For a current
    Ranked map missing from this training snapshot, append a synthetic map
    embedding equal to the mean embedding of trained maps from the same mode.
    The empirical matchup DB already falls back from map-level to mode-level
    statistics, so the whole stack now degrades gracefully instead of silently
    pretending the new map is the first map in the model vocabulary.
    """
    original_maps = list(model.maps)
    if not original_maps or not MAP_MODES:
        return set()

    map_to_mode: dict[str, str] = {}
    for mode, names in OBSERVED_MAP_MODES.items():
        for name in names:
            map_to_mode[name] = mode
    for mode, names in MAP_MODES.items():
        for name in names:
            map_to_mode.setdefault(name, mode)

    rows = [model.m_map]
    names = list(original_maps)
    added: set[str] = set()
    global_mean = model.m_map.mean(axis=0).astype(model.m_map.dtype, copy=False)

    for mode, current_maps in MAP_MODES.items():
        same_mode_indices = [
            i for i, name in enumerate(original_maps) if map_to_mode.get(name) == mode
        ]
        if same_mode_indices:
            fallback = model.m_map[same_mode_indices].mean(axis=0)
            fallback = fallback.astype(model.m_map.dtype, copy=False)
        else:
            fallback = global_mean

        for map_name in current_maps:
            if map_name in names:
                continue
            names.append(map_name)
            rows.append(fallback.reshape(1, -1))
            added.add(map_name)

    if added:
        model.maps = names
        model.m_map = np.vstack(rows).astype(model.m_map.dtype, copy=False)
    return added

if isinstance(TRAINED, FFMInference):
    APPROXIMATE_MAPS = _augment_ffm_with_ranked_maps(TRAINED)
    APPROXIMATE_BRAWLERS = augment_ffm_with_roster(TRAINED, ROSTER_VOCAB)
    EVALUATOR = FFMEvaluator(TRAINED)
    EFFECTIVE_VOCAB = sorted(TRAINED.vocab)
else:
    # Current upstream season configs use the FFM. Legacy plain-FM snapshots
    # cannot safely gain feature columns for newly released brawlers. Keep the
    # trained vocabulary and ask the user to refresh the model if needed.
    EVALUATOR = FMEvaluator(TRAINED)
    EFFECTIVE_VOCAB = sorted(NATIVE_MODEL_BRAWLERS)
    APPROXIMATE_BRAWLERS = set(ROSTER_VOCAB) - NATIVE_MODEL_BRAWLERS

DB = MatchupDB.load(Path(STATE["db_path"]))
SEARCH_LOCK = threading.Lock()

FIRST_TEAM_SLOTS = {1, 4, 5}
SECOND_TEAM_SLOTS = {2, 3, 6}

MODE_LABELS = {
    "bounty": "Bounty",
    "brawlBall": "Brawl Ball",
    "gemGrab": "Gem Grab",
    "heist": "Heist",
    "hotZone": "Hot Zone",
    "knockout": "Knockout",
}


def pct_to_skill(percentile: float) -> float:
    p = min(0.99, max(0.01, percentile / 100.0))
    return math.log(p / (1.0 - p))


def clean_brawler(value: object) -> str:
    return str(value or "").strip().upper()


def slot_side(slot: int) -> str:
    return "first" if slot in FIRST_TEAM_SLOTS else "second"


def parse_slots(raw: object) -> list[str | None]:
    if not isinstance(raw, list) or len(raw) != 6:
        raise ValueError("slots must be a 6-item list.")
    out: list[str | None] = []
    for value in raw:
        b = clean_brawler(value)
        out.append(b or None)
    return out


def validate_sequential(slots: list[str | None]) -> int:
    first_empty = 7
    for i, b in enumerate(slots, start=1):
        if b is None:
            first_empty = i
            break
    if first_empty <= 6 and any(x is not None for x in slots[first_empty:]):
        raise ValueError("Enter picks in draft order; there is a gap in the pick sequence.")
    return first_empty


def team_lists(slots: list[str | None], personal_slot: int) -> tuple[list[str], list[str], bool]:
    user_is_first = personal_slot in FIRST_TEAM_SLOTS
    first = [slots[i - 1] for i in (1, 4, 5) if slots[i - 1]]
    second = [slots[i - 1] for i in (2, 3, 6) if slots[i - 1]]
    if user_is_first:
        return first, second, True
    return second, first, False


def candidate_reasons(
    brawler: str,
    my_picks: list[str],
    opp_picks: list[str],
    mode: str,
    map_name: str,
    skill_ns: float,
    search_pick: dict,
) -> list[str]:
    tier = (
        skill_ns_to_tier(skill_ns, DB.skill_tier_boundaries)
        if DB.skill_tier_boundaries is not None
        else 1
    )

    candidates: list[tuple[float, str]] = []

    if brawler in APPROXIMATE_BRAWLERS:
        candidates.append(
            (2.0, "Limited-data fallback: this brawler is newer than the trained snapshot")
        )

    # Historical evidence shown here is diagnostic context.
    # Bad historical performance should not be hidden from the user.
    stat = DB.brawler_lookup(brawler, mode, map_name, tier)
    if stat and stat.get("n", 0) >= 10:
        wr = float(stat["win_rate"])
        strength = abs(wr - 0.5)
        level = int(stat.get("level", 3))

        if level == 0:
            context = "on this map"
        elif level <= 2:
            context = f"in {MODE_LABELS.get(mode, mode)}"
        else:
            context = "overall"

        if wr < 0.45:
            candidates.append(
                (
                    1.15 + strength,
                    f"Warning: {wr:.0%} historical win rate {context}",
                )
            )
        else:
            candidates.append(
                (
                    0.55 + strength,
                    f"Historical win rate: {wr:.0%} {context}",
                )
            )

    for opp in opp_picks:
        rec = DB.counter_lookup(brawler, opp, mode, map_name, tier)
        if rec and rec.get("n", 0) >= 10:
            wr = float(rec["win_rate"])
            if wr >= 0.52:
                candidates.append(
                    (0.7 + (wr - 0.5), f"Historical matchup into {opp.title()}: {wr:.0%}")
                )

    for mate in my_picks:
        rec = DB.synergy_lookup(brawler, mate, mode, tier)
        if rec and rec.get("n", 0) >= 10:
            delta = float(rec.get("synergy_delta", 0.0))
            if delta > 0.0:
                candidates.append(
                    (0.55 + delta, f"Historical synergy with {mate.title()}: +{delta:.1%}")
                )

    rel = float(search_pick.get("relative_visit_fraction", 0.0) or 0.0)
    if rel >= 1.5:
        candidates.append(
            (0.45, f"MCTS explored this {rel:.1f}× more than a uniform pick")
        )

    if not candidates:
        candidates.append(
            (0.1, "Search prefers its map fit and resilience to likely future replies")
        )

    candidates.sort(key=lambda x: x[0], reverse=True)
    seen: set[str] = set()
    reasons: list[str] = []
    for _, reason in candidates:
        if reason not in seen:
            seen.add(reason)
            reasons.append(reason)
        if len(reasons) == 2:
            break
    return reasons


def make_recommendation(payload: dict) -> dict:
    mode = str(payload.get("mode", "")).strip()
    map_name = str(payload.get("map", "")).strip()
    personal_slot = int(payload.get("personal_slot", 0))
    skill_percentile = float(payload.get("skill_percentile", 75))
    n_simulations = int(payload.get("n_simulations", 20000))

    if personal_slot not in range(1, 7):
        raise ValueError("personal_slot must be from 1 to 6.")

    valid_budgets = {5000, 10000, 20000, 40000, 80000}
    if n_simulations not in valid_budgets:
        raise ValueError(f"n_simulations must be one of {sorted(valid_budgets)}.")

    if mode not in MAP_MODES:
        raise ValueError("Unknown game mode.")
    if map_name not in MAP_MODES[mode]:
        raise ValueError("That map does not match the selected mode.")
    if not isinstance(TRAINED, FFMInference) and map_name not in NATIVE_MODEL_MAPS:
        raise ValueError(
            "This map is newer than the installed model. Run setup.command to update the engine."
        )

    bans_raw = payload.get("bans", [])
    if not isinstance(bans_raw, list):
        raise ValueError("bans must be a list.")
    bans = [clean_brawler(x) for x in bans_raw if clean_brawler(x)]
    if not 3 <= len(bans) <= 6:
        raise ValueError("Enter between 3 and 6 bans.")
    if len(set(bans)) != len(bans):
        raise ValueError("A brawler is banned more than once.")

    slots = parse_slots(payload.get("slots"))
    next_slot = validate_sequential(slots)
    used = [x for x in slots if x]
    if len(set(used)) != len(used):
        raise ValueError("A brawler is picked more than once.")
    overlap = set(used) & set(bans)
    if overlap:
        raise ValueError(f"Picked brawler is banned: {', '.join(sorted(overlap))}")

    vocab = set(EFFECTIVE_VOCAB)
    unknown = (set(used) | set(bans)) - vocab
    if unknown:
        raise ValueError(f"Unknown brawler(s): {', '.join(sorted(unknown))}")

    if next_slot == 7:
        return {
            "status": "complete",
            "message": "Draft complete.",
            "next_slot": None,
            "recommendations": [],
        }

    if next_slot != personal_slot:
        relation = "before" if next_slot < personal_slot else "after"
        if relation == "before":
            return {
                "status": "waiting",
                "message": f"Enter pick {next_slot}. Your pick is slot {personal_slot}.",
                "next_slot": next_slot,
                "recommendations": [],
            }
        return {
            "status": "picked",
            "message": "Your pick has already been entered. You can continue recording the draft.",
            "next_slot": next_slot,
            "recommendations": [],
        }

    my_picks, opp_picks, is_first_pick = team_lists(slots, personal_slot)
    skill_ns = pct_to_skill(skill_percentile)

    # Make identical draft states reproducible.
    # n_simulations is intentionally excluded so larger searches extend
    # the same initial random sequence.
    seed_material = json.dumps(
        {
            "mode": mode,
            "map": map_name,
            "personal_slot": personal_slot,
            "skill_percentile": skill_percentile,
            "bans": sorted(bans),
            "slots": slots,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")

    search_seed = int.from_bytes(
        hashlib.blake2b(seed_material, digest_size=8).digest(),
        "little",
    )

    with SEARCH_LOCK:
        result = recommend(
            my_picks=my_picks,
            opp_picks=opp_picks,
            mode=mode,
            map_name=map_name,
            skill_ns=skill_ns,
            is_first_pick=is_first_pick,
            bans=bans,
            n_simulations=n_simulations,
            n_top=3,
            min_counter_games=10,
            evaluator=EVALUATOR,
            db=DB,
            rng=np.random.default_rng(search_seed),
        )

    recs = []
    for rank, pick in enumerate(result.top_picks[:3], start=1):
        b = pick["brawler"]
        recs.append(
            {
                "rank": rank,
                "brawler": b,
                "estimated_win_prob": float(pick.get("estimated_win_prob", 0.0)),
                "win_prob_delta": float(pick.get("win_prob_delta", 0.0)),
                "visit_fraction": float(pick.get("visit_fraction", 0.0)),
                "relative_visit_fraction": float(
                    pick.get("relative_visit_fraction", 0.0)
                ),
                "data_support": (
                    "roster_fallback" if b in APPROXIMATE_BRAWLERS else "trained"
                ),
                "reasons": candidate_reasons(
                    b, my_picks, opp_picks, mode, map_name, skill_ns, pick
                ),
            }
        )

    return {
        "status": "recommend",
        "message": f"Your turn — pick {personal_slot}.",
        "next_slot": next_slot,
        "recommendations": recs,
        "elapsed_sec": float(result.elapsed_sec),
        "simulations": n_simulations,
        "dataset_rows": STATE.get("dataset_rows"),
        "map_support": "mode_fallback" if map_name in APPROXIMATE_MAPS else "map_specific",
        "roster_fallback_count": len(APPROXIMATE_BRAWLERS),
    }


class Handler(BaseHTTPRequestHandler):
    server_version = "BrawlDraftLocal/1.3"

    def log_message(self, fmt: str, *args) -> None:
        print("[local]", fmt % args)

    def send_json(self, data: dict, status: int = 200) -> None:
        blob = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(blob)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(blob)

    def do_GET(self) -> None:
        path = unquote(urlparse(self.path).path)
        if path == "/api/meta":
            self.send_json(
                {
                    "vocab": EFFECTIVE_VOCAB,
                    "model_vocab": sorted(NATIVE_MODEL_BRAWLERS),
                    "approximate_brawlers": sorted(APPROXIMATE_BRAWLERS),
                    "roster": ROSTER_META,
                    "model_snapshot": STATE.get("model_config") or Path(STATE.get("config_path", "")).stem or None,
                    "map_modes": MAP_MODES,
                    "dataset_rows": STATE.get("dataset_rows"),
                    "native_model_maps": sorted(NATIVE_MODEL_MAPS),
                    "approximate_maps": sorted(APPROXIMATE_MAPS),
                    "map_catalog": {
                        "scope": MAP_CATALOG.get("scope"),
                        "season": MAP_CATALOG.get("season"),
                        "featured_mode": MAP_CATALOG.get("featured_mode"),
                        "verified_on": MAP_CATALOG.get("verified_on"),
                    },
                }
            )
            return
        if path == "/api/health":
            self.send_json({"ok": True})
            return

        if path == "/":
            file_path = STATIC / "index.html"
        else:
            file_path = (STATIC / path.lstrip("/")).resolve()
            if STATIC.resolve() not in file_path.parents:
                self.send_error(HTTPStatus.NOT_FOUND)
                return

        if not file_path.exists() or not file_path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return

        data = file_path.read_bytes()
        mime, _ = mimetypes.guess_type(str(file_path))
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", (mime or "application/octet-stream") + ("; charset=utf-8" if (mime or "").startswith("text/") else ""))
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self) -> None:
        if urlparse(self.path).path != "/api/recommend":
            self.send_error(HTTPStatus.NOT_FOUND)
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length) or b"{}")
            result = make_recommendation(payload)
            self.send_json(result)
        except (ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
            self.send_json({"error": str(exc)}, status=400)
        except Exception as exc:
            print("Recommendation error:", repr(exc))
            self.send_json(
                {
                    "error": "The local engine hit an unexpected error.",
                    "detail": str(exc),
                },
                status=500,
            )


def open_browser() -> None:
    webbrowser.open(f"http://{HOST}:{PORT}")


if __name__ == "__main__":
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    print("")
    print("Brawl Draft Local is running.")
    print(f"Open: http://{HOST}:{PORT}")
    print("Press Control-C in this window to stop it.")
    print("")
    threading.Timer(0.7, open_browser).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping.")
    finally:
        server.server_close()
