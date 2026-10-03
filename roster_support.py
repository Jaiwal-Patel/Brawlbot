from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np


def _normalise_name(value: object) -> str:
    return str(value or "").strip().upper()


def load_roster_catalog(path: Path, fallback_vocab: Iterable[str]) -> tuple[list[str], dict]:
    """Load the cached current roster, falling back to the trained vocabulary.

    The roster cache is deliberately separate from the model vocabulary so a
    newly released brawler can be entered immediately without pretending that
    the trained snapshot already contains real statistics for it.
    """
    fallback = sorted({_normalise_name(x) for x in fallback_vocab if _normalise_name(x)})
    if not path.exists():
        return fallback, {
            "source": "model_vocab_fallback",
            "updated_at": None,
            "count": len(fallback),
        }

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        raw = payload.get("brawlers", [])
        roster = sorted({_normalise_name(x) for x in raw if _normalise_name(x)})
        if not roster:
            raise ValueError("roster cache contains no brawlers")
        meta = {
            "source": payload.get("source", "cached_roster"),
            "updated_at": payload.get("updated_at"),
            "count": len(roster),
        }
        return roster, meta
    except Exception as exc:
        return fallback, {
            "source": "model_vocab_fallback",
            "updated_at": None,
            "count": len(fallback),
            "warning": f"Could not read roster cache: {exc}",
        }


def augment_ffm_with_roster(model, roster_names: Iterable[str]) -> set[str]:
    """Append neutral roster-average rows for brawlers absent from an FFM snapshot.

    This does *not* invent map, synergy, or counter evidence. It only prevents an
    unknown brawler from aliasing to vocabulary index 0 inside the upstream
    evaluator. The matchup database naturally returns neutral fallbacks when no
    observations exist. A later model refresh replaces these neutral rows with
    learned parameters once the brawler is present in the training dataset.
    """
    original = list(model.vocab)
    existing = set(original)
    added = sorted({_normalise_name(x) for x in roster_names if _normalise_name(x)} - existing)
    if not added:
        return set()

    n_new = len(added)

    # An average learned brawler is a less arbitrary prior than either mapping an
    # unknown name to index 0 or zeroing every field. Keep each parameter family
    # in its own learned scale and use the roster mean as the neutral prior.
    w_mean = np.asarray(model.w).mean(dtype=np.float64).astype(model.w.dtype)
    model.w = np.concatenate(
        [model.w, np.full(n_new, w_mean, dtype=model.w.dtype)]
    )

    for attr in ("e_syn", "e_att", "e_def", "e_ctx"):
        arr = getattr(model, attr)
        mean_row = arr.mean(axis=0, dtype=np.float64).astype(arr.dtype, copy=False)
        extra = np.repeat(mean_row.reshape(1, -1), n_new, axis=0)
        setattr(model, attr, np.vstack([arr, extra]).astype(arr.dtype, copy=False))

    model.vocab = original + added
    return set(added)
