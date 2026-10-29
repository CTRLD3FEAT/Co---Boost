# Co-Boost dataset notes

## Date: 2026-10-23 — Non-food commodities removed from analytics

**Decision by TEAM_NRZB: Betel Leaves and Arecanuts are excluded from the Co-Boost
food-security dataset.** They are chewed stimulants (betel-vine leaves / areca nut),
not food consumables, so tracking them as "staple food prices" distorts
food-inflation monitoring, anomaly scores and spike predictions.

Removed commodity variants (6, 1,908 rows of the 36,008-row wrp dataset, ~5%):

- Betel Leaves (Average), Betel Leaves (Medium), Betel Leaves (Small)
- Arecanuts (Average), Arecanuts (Medium), Arecanuts (Small)

### How this is implemented

- The raw extraction `outputs/official_prices_wrp.csv` is **kept unchanged**
  (auditability: every record traceable to its source PDF).
- `outputs/commodity_exclusions.csv` is the single machine-readable filter table
  (pattern + reason). It is applied inside `baseline.load_prices()`, so every
  consumer — baselines, backtests, ML models and both dashboards — transparently
  drops these commodities.
- If you disagree with an exclusion, delete/edit the row in
  `commodity_exclusions.csv` and it comes back everywhere.

Commodity count in the canonical view: 171 → 165.

## Date: 2026-10-23 — Name normalization + corrupted-row removal

**Decision by TEAM_NRZB:**

### 1. Unicode dash variants merged (not excluded)
The DCS PDFs print item names with two different dash glyphs (ASCII '-' and
Unicode '‐' U+2010). The same commodity therefore appeared as two separate
series, e.g. `Beans - Butter` vs `Beans ‐ Butter`, splitting each commodity's
history and forecast into two halves.

- Fix applied in `baseline.load_prices()` via `_normalize_name()`: all dash
  glyphs unified to ASCII '-', whitespace collapsed; applied to commodity,
  parent and child columns.
- Result: 18 variant-name families merged (e.g. Beans, Mysore Dhall, Fresh
  Fish, Nadu, Raw Red/White, Samba...). **No rows were lost** — history for
  both variants now belongs to one canonical series (verified: `Beans - Butter`
  = 288 continuous weeks 2019-12 → 2026-10).

### 2. Corrupted row excluded
`Salt 1Kg.Pkt. 60.00` (DCS-WRP-2021-10-W3.pdf) is a parsing artifact — the
price 60.00 got glued into the commodity name. A clean weekly **Salt** series
already exists, so the row is excluded, not repaired, and Salt tracking is
unaffected (`outputs/commodity_exclusions.csv` lists it with reason).

Commodity count in the canonical view: 165 → **146**.
Rows: 34,100 → 34,099.

### If the corrupted row ever needs recovery
The true name/price is recoverable from the source PDF
(`Dataset/DCS-WRP-2021-10-W3.pdf`) or by hardening the name-splitting heuristic
in `Scripts/parse_wrp.py`; we chose exclusion because no other week shows the
bug and Salt is fully covered by the clean series.

## 2026-10-28: week-label authority fixed

Extract reveals DCS sometimes prints stale copy-paste headlines
(e.g. DCS-WRP-2024-01-W4.pdf headline says "4th Week of January 2023",
while its own % column proves Jan-2024). parse_wrp.py now treats the
file name as the canonical reporting week and logs any TITLE/FN MISMATCH
in the extraction notes. Re-extraction: 320/320 files, dup keys in the
merged view = 0; mislabeled weeks restored (Jan-2024-W4 re-attached to
its true week). Backtest + spike-ML leaderboards refreshed against it.
