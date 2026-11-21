# V3.3 evaluation pool registry

This is a **new** registry; it does not overwrite the historical
`docs/riichiformer/EVAL_POOL_REGISTRY.md`.

## Consumed (do not reuse for confirm)

`[10000,13000)`, `[10000,10500)` (gate), `[20000,23000)`, `[30000,33000)`,
`[40000,40030)` (akochan trend), `[50000,53000)`, `[60000,63002)` (akochan),
`[70000,73000)`, `[120000,126000)`, EX1 `900000000+`/`901000000+`/`902000000+`,
and the akochan single point `40010`. See
`riichiformer/validation/disjoint.py` for the machine-readable list.
The B0 train sample is 500 human games selected by `sha256("20261121:"+game_id)`;
it is train-only and not an evaluation pool.

## V3.3 smoke pool (proposed, not yet consumed)

- `seed_key`: generated from `secrets.randbits(64)` and recorded in the run manifest;
  never a hard-coded public example.
- `seed_window`: `[200000, 200100)` was proposed but **not** the actual lane run.
- Actual V3.3 smoke/debug windows (all role `smoke`, never confirm):
  `[220000,220004)`, `[221000,221004)`, `[222000,222004)` (A tests),
  `[241000,241004)` and `[241004,241008)` (A smoke),
  `[250000,250025)` (Lead 100-game A0 self-control; 25 sets; key deliberately
  pinned to `0x9E3779B97F4A7C15` for this one gate, not a generated/confirm key),
  `[300000,300025)` (B 100-game validation/diagnostic corpus).
- None of these is a `confirm` pool and none may be relabeled as one.
- No `confirm` pool is created or consumed in this wave. A future confirm pool must be
  audited across the whole repository and frozen before its first game.
