# V3.3 evaluation pool registry

This is a **new** registry; it does not overwrite the historical
`docs/riichiformer/EVAL_POOL_REGISTRY.md`.

## Consumed (do not reuse for confirm)

`[10000,13000)`, `[20000,23000)`, `[30000,33000)`, `[50000,53000)`,
`[70000,73000)`, `[120000,126000)`, and the akochan single point `40010`.
The B0 train sample is 500 human games selected by `sha256("20261121:"+game_id)`;
it is train-only and not an evaluation pool.

## V3.3 smoke pool (proposed, not yet consumed)

- `seed_key`: generated from `secrets.randbits(64)` and recorded in the run manifest;
  never a hard-coded public example.
- `seed_window`: `[200000, 200100)` for the CPU smoke (max 100 sets), role `smoke`.
- No `confirm` pool is created or consumed in this wave. A future confirm pool must be
  audited across the whole repository and frozen before its first game.
