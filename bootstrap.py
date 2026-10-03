from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

from bsdraft.cli.train import resolve_source
from bsdraft.config import load_config
from bsdraft.data.prep import build_game_dataset
from bsdraft.data.matchup_db import (
    MatchupDB,
    assign_skill_tiers,
    build_brawler_stats,
    build_counter_matrix,
    build_synergy_matrix,
)


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("usage: bootstrap.py CONFIG_PATH STATE_JSON")

    config_path = Path(sys.argv[1])
    state_path = Path(sys.argv[2]).resolve()

    cfg = load_config(config_path)
    source = resolve_source(cfg)

    print(f"Loading dataset: {source}")
    df, _ = build_game_dataset(
        source,
        elo_min=cfg.data.elo_min,
        elo_max=cfg.data.elo_max,
    )

    if df.empty:
        raise RuntimeError("Resolved dataset is empty.")

    # Keep the real map→mode relation for the UI.
    pairs = (
        df[["mode", "map"]]
        .dropna()
        .drop_duplicates()
        .sort_values(["mode", "map"])
    )
    map_modes: dict[str, list[str]] = {}
    for row in pairs.itertuples(index=False):
        map_modes.setdefault(str(row.mode), []).append(str(row.map))

    print("Building matchup database...")
    tiered, boundaries = assign_skill_tiers(df)
    brawler = build_brawler_stats(tiered)
    counter = build_counter_matrix(tiered)
    synergy = build_synergy_matrix(tiered, brawler)
    db = MatchupDB(
        brawler=brawler,
        counter=counter,
        synergy=synergy,
        skill_tier_boundaries=boundaries,
    )

    engine_root = Path.cwd().resolve()
    db_path = engine_root / "data" / "brawl_draft_local_matchup_db.pkl"
    db.save(db_path)

    model_path = (Path(cfg.run_dir) / "fm_model.pkl").resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"Trained model not found at {model_path}")

    with model_path.open("rb") as f:
        trained = pickle.load(f)

    vocab = list(getattr(trained, "vocab", ()))
    maps = list(getattr(trained, "maps", ()))
    modes = list(getattr(trained, "modes", ()))
    if not vocab:
        schema = getattr(trained, "schema", None)
        if schema is not None:
            vocab = list(schema.vocab)
            maps = list(schema.maps)
            modes = list(schema.modes)

    state = {
        "model_path": str(model_path),
        "db_path": str(db_path.resolve()),
        "config_path": str(config_path.resolve()),
        "model_config": config_path.stem,
        "engine_root": str(engine_root),
        "vocab": sorted(vocab),
        "model_vocab_count": len(vocab),
        "maps": sorted(maps),
        "modes": sorted(modes),
        "map_modes": map_modes,
        "observed_map_modes": map_modes,
        "skill_boundaries": list(boundaries),
        "dataset_rows": int(len(df)),
    }
    state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
    print(f"Wrote local app state → {state_path}")
    print(f"Rows available to local stats layer: {len(df):,}")


if __name__ == "__main__":
    main()
