# Brawl Draft Local v1.4

A Mac-local Brawl Stars Ranked draft assistant built around the MIT-licensed
`elixf7/brawlstars-draft-agent` statistical engine.

## What it does

- Choose the **mode** and **map** from the current Ranked pool.
- Enter **3–6 bans**.
- Choose your personal global draft slot, **1–6**.
- Record the other picks as they happen.
- When your slot arrives, the app runs Monte Carlo Tree Search locally and returns
  **exactly 3 alternatives**.
- Each recommendation shows an estimated draft win probability plus short
  counter / synergy / map-data reasons when those signals are available.
- The map catalogue is independent of the training snapshot, so a newer Ranked
  map can use a conservative mode-level fallback.
- The brawler roster is also independent of the training snapshot. The app can
  recognize newly released brawlers immediately instead of being frozen to the
  season used to train the model.

The six global draft slots are modeled as:

`A → B → B → A → A → B`

So slots `1, 4, 5` are on the first-pick team and slots `2, 3, 6` are on the
second-pick team.

## First-time setup on macOS

1. Unzip this folder somewhere permanent, such as `~/Applications/BrawlDraftLocal`.
2. Double-click **`setup.command`**.
3. macOS may ask whether Terminal can run it. Allow it.
4. Setup installs the Python environment, downloads the current draft engine,
   chooses the newest season config in that engine, trains the compact FFM model,
   builds matchup/counter/synergy data, refreshes the released-brawler roster,
   and runs a smoke test.

If Finder says the `.command` file is not executable, open Terminal in this folder
and run:

```bash
chmod +x setup.command run.command update_roster.command
./setup.command
```

## Run it

After setup, double-click **`run.command`**.

The browser opens at:

`http://127.0.0.1:8765`

The server binds only to `127.0.0.1`, so it is not exposed to other devices on
your network.

Before the local server starts, `run.command` checks the released-brawler roster.
At most once per 24 hours it refreshes that small roster file from BrawlAPI. If
there is no internet connection, the last cached roster is kept. There is **no
cloud call during a live draft**.

## Current roster handling

The current released roster is fetched from the unauthenticated static endpoint:

`https://api.brawlapi.com/v1/brawlers`

The roster cache is `brawler_roster.json` and is separate from the model.

If a brawler exists in the current roster but was not present in the model's
training snapshot, the app does **not** alias that brawler to some unrelated old
brawler. For the FFM it appends a neutral, roster-average parameter row. The
matchup database then naturally falls back to neutral values where no real
map/counter/synergy observations exist. The UI labels recommendations involving
such a brawler as **limited data**.

This is intentionally conservative. A roster refresh makes a new brawler usable;
it does not pretend that historical evidence exists. To learn real statistics
for new brawlers, rerun `setup.command` later when the upstream season dataset has
collected them.

You can force only the roster refresh, without retraining, by double-clicking:

`update_roster.command`

## Ranked map catalogue

The app ships with a separate `ranked_maps.json` catalogue so the map dropdown is
not limited to maps present in the training dataset. If a selected map was not
seen by the FFM, the app uses the mean map embedding for trained maps in that
mode, while matchup lookups fall back to mode-level/global evidence.

Changing only `ranked_maps.json` does **not** require retraining.

## What the header means

v1.2 no longer labels the whole app with the map-pool season number. The top
badge now shows:

`match rows · current brawler count · current map count · local`

The smaller data-support line separately shows the actual model config (for
example `season53`) and how many current brawlers, if any, are using the
limited-data fallback.

## Search speed

The choices are:

- 5,000 — Quick
- 10,000 — Fast
- 20,000 — Balanced
- 40,000 — Strong
- 80,000 — Max

Balanced (20,000 simulations) is the default. Higher budgets reduce Monte Carlo
search noise when many brawlers remain available.

Searches are deterministically seeded from the draft state, so identical drafts
produce stable results at the same search strength.

Historical win-rate, counter, and synergy lines shown on recommendation cards
are diagnostic context rather than direct MCTS weights. Poor supported
historical win rates are shown as warnings instead of being hidden.

## Updating the statistical model

Run `setup.command` again. It downloads the current upstream source, selects the
newest `configs/season*.toml`, retrains the model, rebuilds matchup data, and
refreshes the roster cache.

## Files

- `app.py` — local HTTP server + recommender adapter
- `roster_support.py` — safe current-roster/model fallback support
- `update_roster.py` — current released-roster refresher
- `update_roster.command` — one-click roster-only refresh
- `ranked_maps.json` — selectable Ranked map catalogue
- `bootstrap.py` — builds local matchup/counter/synergy data and app state
- `smoke_test.py` — verifies the wrapper and three-pick recommendation path
- `static/index.html` — local draft UI
- `setup.command` — first-time/model-update setup
- `run.command` — launcher

## Privacy

During a live draft, recommendation requests go only from your browser to the
local server on your Mac. Setup/model updates and the pre-launch roster refresh
use the internet.

## Attribution

The recommendation engine is from `elixf7/brawlstars-draft-agent` and is
published under the MIT License. Brawler reference data is refreshed from
BrawlAPI/Brawlify's public static reference endpoint.

Not affiliated with, endorsed by, or sponsored by Supercell.
Brawl Stars is a trademark of Supercell Oy.


## v1.4 — Ban advice, full draft, final evaluation

- Three empirical ban suggestions appear before picks begin.
- Recommendations appear for every allied pick slot.
- Three opponent-pick predictions appear on enemy turns.
- Suggestions fill the pick field; Record pick confirms them.
- After all six picks, the terminal model evaluates the completed draft.
- Win estimates are currently uncalibrated.
- Ban advice is an empirical map-strength heuristic, not ban MCTS.
- No model retraining is required for these interface/backend features.
