from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np

from bsdraft.data.matchup_db import MatchupDB
from bsdraft.fm.ffm import FFMInference
from bsdraft.mcts.ffm_evaluator import FFMEvaluator
from bsdraft.mcts.evaluator import FMEvaluator
from bsdraft.mcts.recommend import recommend

from roster_support import augment_ffm_with_roster, load_roster_catalog

ROOT = Path(__file__).resolve().parent
state = json.loads((ROOT / "agent_state.json").read_text(encoding="utf-8"))
catalog = json.loads((ROOT / "ranked_maps.json").read_text(encoding="utf-8"))
map_modes = catalog["map_modes"]

assert sum(len(v) for v in map_modes.values()) == 26, "Expected 26 current Ranked maps."
assert "In the Liminal" in map_modes["hotZone"]
assert "Quick Travel" in map_modes["hotZone"]

with open(state["model_path"], "rb") as f:
    trained = pickle.load(f)

native_vocab = list(getattr(trained, "vocab", state.get("vocab", [])))
roster, roster_meta = load_roster_catalog(ROOT / "brawler_roster.json", native_vocab)

# Test the exact case that motivated the map-catalogue fix. If this map is newer
# than the model snapshot, give it the same neutral mode-average fallback used by
# app.py instead of allowing the upstream evaluator to alias it to map index 0.
mode = "hotZone"
map_name = "In the Liminal"

if isinstance(trained, FFMInference):
    if map_name not in trained.maps:
        observed = state.get("observed_map_modes", state.get("map_modes", {}))
        trained_hot_zone = set(observed.get(mode, []))
        idx = [i for i, name in enumerate(trained.maps) if name in trained_hot_zone]
        if idx:
            fallback = trained.m_map[idx].mean(axis=0)
        else:
            fallback = trained.m_map.mean(axis=0)
        trained.maps = list(trained.maps) + [map_name]
        trained.m_map = np.vstack(
            [trained.m_map, fallback.astype(trained.m_map.dtype, copy=False).reshape(1, -1)]
        ).astype(trained.m_map.dtype, copy=False)
    added_brawlers = augment_ffm_with_roster(trained, roster)
    assert len(trained.vocab) == len(trained.w) == len(trained.e_syn)
    assert set(roster).issubset(set(trained.vocab))
    evaluator = FFMEvaluator(trained)
else:
    # Legacy plain-FM snapshots cannot gain a new one-hot map column safely.
    # Smoke-test a native map instead; app.py asks the user to retrain for those.
    evaluator = FMEvaluator(trained)
    mode = next(iter(state["map_modes"]))
    map_name = state["map_modes"][mode][0]

db = MatchupDB.load(Path(state["db_path"]))
bans = list(evaluator._fm.schema.vocab)[:3]

result = recommend(
    my_picks=[],
    opp_picks=[],
    mode=mode,
    map_name=map_name,
    skill_ns=0.0,
    is_first_pick=True,
    bans=bans,
    n_simulations=100,
    n_top=3,
    min_counter_games=10,
    evaluator=evaluator,
    db=db,
    rng=np.random.default_rng(42),
)

assert len(result.top_picks) == 3
assert all(p["brawler"] not in bans for p in result.top_picks)
print(
    "Smoke test OK:",
    f"roster={len(roster)} ({roster_meta.get('source')})",
    mode,
    "/",
    map_name,
    "→",
    ", ".join(p["brawler"] for p in result.top_picks),
    f"({result.elapsed_sec:.2f}s for 100 sims)",
)
