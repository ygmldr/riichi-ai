# Era / rule drift report (PR-07R)

- generated: 2026-11-20T21:17:55Z
- elapsed: 812.9 s
- command: `/home/ygsldr/riichi-ai/.venv/bin/python scripts/rf_drift.py --manifest /home/ygsldr/riichi-ai/artifacts/riichiformer/audit/source_audit.parquet --out /home/ygsldr/riichi-ai/artifacts/riichiformer/drift --report docs/riichiformer/DRIFT_REPORT.md --per-year 400 --seed 20261121 --workers 8 --years 2019-2026`
- working directory: `/home/ygsldr/riichi-ai/worktrees/h05-drift`
- seed: `20261121`; per-year cap: `400`; year filter: `[2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026]`; workers: `8`
- sample: 3200 parsed / 3200 selected (0 failed)
- replay: ENABLED; replayed games: 3200; replay errors: 0
- manifest: `/home/ygsldr/riichi-ai/artifacts/riichiformer/audit/source_audit.parquet` (sha256 `ebb90909558e16e57a221b0e773c923a5d19a3df84dc48806d13094b16543f52`)
- platform metadata: inventory sources `{'tenhou_old': 1600, 'tenhou_recent': 1600}`; per-log rule flags are **UNKNOWN / not verified**, so this report does not claim a ruleset id
- python: `3.12.3`
- libriichi: `/home/ygsldr/riichi-ai/vendor/mortal/target/release/libriichi.so` (sha256 `fc05371f8035283ce61147e440c82eac16506cbb45584410607746afcbd91eba`)
- PYTHONPATH: `/home/ygsldr/riichi-ai/vendor/mortal/target/release`

## Sample and failures

Failures are counted by stage and reason; no selected game is silently dropped.

| year | selected | parsed | failed |
|---:|---:|---:|---:|
| 2019 | 400 | 400 | 0 |
| 2020 | 400 | 400 | 0 |
| 2021 | 400 | 400 | 0 |
| 2022 | 400 | 400 | 0 |
| 2023 | 400 | 400 | 0 |
| 2024 | 400 | 400 | 0 |
| 2025 | 400 | 400 | 0 |
| 2026 | 400 | 400 | 0 |

Parse/read failures: none.

Replay failures: none.

## Pre-registered windows

| window | games | years | months | kyoku | decisions | note |
|---|---:|---|---:|---:|---:|---|
| `2019_2022` | 1600 | [2019, 2020, 2021, 2022] | 48 | 17013 | 1055818 |  |
| `2023_2024` | 800 | [2023, 2024] | 24 | 8424 | 519147 |  |
| `2025` | 400 | [2025] | 12 | 4317 | 266983 |  |
| `2026` | 400 | [2026] | 1 | 4261 | 264122 | 2026 is January-only locally |

## Window action/length metrics

Rates are per kyoku. `dahai`, `chi`, `pon`, `kan` and `ron` are event counts; `kan` includes daiminkan/ankan/kakan. `decisions` requires replay.

| window | games | kyoku | decisions | events p50/p90/p99 | dahai/kyoku | chi/kyoku | pon/kyoku | kan/kyoku | ron/kyoku |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|
| `2019_2022` | 1600 | 17013 | 1055818 | 1073/1442/1851 | 48.3444 | 0.8195 | 1.1117 | 0.0814 | 0.5008 |
| `2023_2024` | 800 | 8424 | 519147 | 1057/1449/1842 | 47.9085 | 0.8012 | 1.1286 | 0.0827 | 0.4975 |
| `2025` | 400 | 4317 | 266983 | 1085/1464/1719 | 48.0656 | 0.8142 | 1.1258 | 0.0760 | 0.5071 |
| `2026` | 400 | 4261 | 264122 | 1056/1495/1801 | 48.2664 | 0.8181 | 1.1253 | 0.0864 | 0.4931 |

## Window rates (95% CI)

CIs are normal approximations for event counts per kyoku (rates may exceed 1) and normal-approximation means for points/deal-ins; they are **sampling** intervals for the selected games, not causal intervals.

| window | reach/kyoku | calls/kyoku | hora/kyoku | ryukyoku/kyoku | deal-ins/game | mean hora pts |
|---|---|---|---|---|---:|---|
| `2019_2022` | 0.7228 [0.7100, 0.7356] | 2.0126 [1.9913, 2.0340] | 0.8388 [0.8251, 0.8526] | 0.1668 [0.1607, 0.1730] | 5.3250 [5.2310, 5.4190] | 6790.4 [6716.8, 6864.0] |
| `2023_2024` | 0.7269 [0.7086, 0.7451] | 2.0125 [1.9822, 2.0428] | 0.8400 [0.8204, 0.8596] | 0.1644 [0.1558, 0.1731] | 5.2387 [5.1017, 5.3758] | 6756.5 [6652.0, 6860.9] |
| `2025` | 0.7287 [0.7033, 0.7542] | 2.0160 [1.9736, 2.0583] | 0.8515 [0.8240, 0.8790] | 0.1547 [0.1430, 0.1665] | 5.4725 [5.2811, 5.6639] | 6775.6 [6631.4, 6919.8] |
| `2026` | 0.7113 [0.6860, 0.7367] | 2.0298 [1.9870, 2.0726] | 0.8442 [0.8166, 0.8718] | 0.1624 [0.1503, 0.1745] | 5.2525 [5.0761, 5.4290] | 6853.9 [6706.4, 7001.3] |

## Game termination

`reached S4/W` is the share of games whose last kyoku is South 4 or a West extension; the remaining games ended early (e.g. bust). Final-event shares describe the last hora/ryukyoku of a game.

| window | reached S4/W | 95% CI | final hora | 95% CI | final ryukyoku | 95% CI |
|---|---:|---|---:|---|---:|---|
| `2019_2022` | 0.7819 | [0.7610, 0.8014] | 0.9225 | [0.9084, 0.9346] | 0.0775 | [0.0654, 0.0916] |
| `2023_2024` | 0.7825 | [0.7526, 0.8097] | 0.9275 | [0.9074, 0.9435] | 0.0725 | [0.0565, 0.0926] |
| `2025` | 0.7925 | [0.7501, 0.8294] | 0.9150 | [0.8836, 0.9385] | 0.0850 | [0.0615, 0.1164] |
| `2026` | 0.7775 | [0.7342, 0.8155] | 0.9125 | [0.8807, 0.9364] | 0.0875 | [0.0636, 0.1193] |

## Per-year basic metrics

| year | games | kyoku | decisions | events p50/p90/p99 | reach/kyoku | calls/kyoku | ron/kyoku |
|---:|---:|---:|---:|---|---:|---:|---:|
| 2019 | 400 | 4261 | 264730 | 1067/1461/1955 | 0.7076 | 2.0408 | 0.5006 |
| 2020 | 400 | 4199 | 261179 | 1053/1416/1788 | 0.7266 | 1.9914 | 0.5111 |
| 2021 | 400 | 4221 | 260539 | 1069/1399/1734 | 0.7114 | 2.0379 | 0.4968 |
| 2022 | 400 | 4332 | 269370 | 1105/1494/1851 | 0.7452 | 1.9808 | 0.4949 |
| 2023 | 400 | 4168 | 256032 | 1042/1401/1834 | 0.7195 | 2.0173 | 0.4950 |
| 2024 | 400 | 4256 | 263115 | 1065/1466/1842 | 0.7340 | 2.0078 | 0.5000 |
| 2025 | 400 | 4317 | 266983 | 1085/1464/1719 | 0.7287 | 2.0160 | 0.5071 |
| 2026 | 400 | 4261 | 264122 | 1056/1495/1801 | 0.7113 | 2.0298 | 0.4931 |

## Conditional strata (unadjusted, not causal)

`drift_strata.csv` contains all three stratum types for every window, including `pre_2019`. Rates below are raw conditional frequencies.

### window x bakaze

| stratum | kyoku | reach/kyoku | 95% CI | calls/kyoku | 95% CI |
|---|---:|---|---|---|---|
| `2019_2022|E` | 9310 | 0.7880 | [0.7699, 0.8060] | 1.9459 | [1.9175, 1.9742] |
| `2019_2022|S` | 7632 | 0.6456 | [0.6275, 0.6636] | 2.0869 | [2.0545, 2.1193] |
| `2019_2022|W` | 71 | 0.4789 | [0.3179, 0.6398] | 2.7887 | [2.4003, 3.1772] |
| `2023_2024|E` | 4602 | 0.8094 | [0.7834, 0.8354] | 1.9500 | [1.9097, 1.9904] |
| `2023_2024|S` | 3777 | 0.6262 | [0.6009, 0.6514] | 2.0855 | [2.0395, 2.1316] |
| `2023_2024|W` | 45 | 0.7333 | [0.4831, 0.9835] | 2.2667 | [1.8268, 2.7066] |
| `2025|E` | 2347 | 0.8087 | [0.7723, 0.8451] | 1.9595 | [1.9029, 2.0162] |
| `2025|S` | 1936 | 0.6384 | [0.6028, 0.6740] | 2.0723 | [2.0082, 2.1364] |
| `2025|W` | 34 | 0.3529 | [0.1532, 0.5526] | 2.7059 | [2.1530, 3.2588] |
| `2026|E` | 2308 | 0.7626 | [0.7269, 0.7982] | 1.9731 | [1.9158, 2.0304] |
| `2026|S` | 1919 | 0.6529 | [0.6168, 0.6891] | 2.0860 | [2.0214, 2.1506] |
| `2026|W` | 34 | 0.5294 | [0.2848, 0.7740] | 2.7059 | [2.1530, 3.2588] |

### window x bakaze x all_last (bakaze=S, kyoku=4)

| stratum | kyoku | reach/kyoku | 95% CI | calls/kyoku | 95% CI |
|---|---:|---|---|---|---|
| `2019_2022|E|other` | 9310 | 0.7880 | [0.7699, 0.8060] | 1.9459 | [1.9175, 1.9742] |
| `2019_2022|S|all_last` | 1545 | 0.4498 | [0.4164, 0.4833] | 2.2110 | [2.1369, 2.2851] |
| `2019_2022|S|other` | 6087 | 0.6953 | [0.6743, 0.7162] | 2.0554 | [2.0193, 2.0914] |
| `2019_2022|W|other` | 71 | 0.4789 | [0.3179, 0.6398] | 2.7887 | [2.4003, 3.1772] |
| `2023_2024|E|other` | 4602 | 0.8094 | [0.7834, 0.8354] | 1.9500 | [1.9097, 1.9904] |
| `2023_2024|S|all_last` | 797 | 0.4667 | [0.4193, 0.5142] | 2.2133 | [2.1100, 2.3166] |
| `2023_2024|S|other` | 2980 | 0.6688 | [0.6394, 0.6982] | 2.0513 | [1.9999, 2.1028] |
| `2023_2024|W|other` | 45 | 0.7333 | [0.4831, 0.9835] | 2.2667 | [1.8268, 2.7066] |
| `2025|E|other` | 2347 | 0.8087 | [0.7723, 0.8451] | 1.9595 | [1.9029, 2.0162] |
| `2025|S|all_last` | 384 | 0.4740 | [0.4051, 0.5428] | 2.2005 | [2.0521, 2.3489] |
| `2025|S|other` | 1552 | 0.6791 | [0.6381, 0.7201] | 2.0406 | [1.9695, 2.1117] |
| `2025|W|other` | 34 | 0.3529 | [0.1532, 0.5526] | 2.7059 | [2.1530, 3.2588] |
| `2026|E|other` | 2308 | 0.7626 | [0.7269, 0.7982] | 1.9731 | [1.9158, 2.0304] |
| `2026|S|all_last` | 401 | 0.4738 | [0.4064, 0.5412] | 2.2968 | [2.1484, 2.4451] |
| `2026|S|other` | 1518 | 0.7003 | [0.6582, 0.7424] | 2.0303 | [1.9586, 2.1020] |
| `2026|W|other` | 34 | 0.5294 | [0.2848, 0.7740] | 2.7059 | [2.1530, 3.2588] |

### window x bakaze x reach turn share

Denominator is all reach declarations in the window/bakaze; the turn is the declaring player's own discard index (junme). This is **not** a per-kyoku rate.

| stratum | reaches | share | 95% CI |
|---|---:|---|---|
| `2019_2022|E|0-5` | 1642 | 0.2238 | [0.2144, 0.2335] |
| `2019_2022|E|6-11` | 4687 | 0.6389 | [0.6278, 0.6498] |
| `2019_2022|E|12+` | 1007 | 0.1373 | [0.1296, 0.1453] |
| `2019_2022|S|0-5` | 1108 | 0.2249 | [0.2134, 0.2368] |
| `2019_2022|S|6-11` | 3135 | 0.6363 | [0.6228, 0.6496] |
| `2019_2022|S|12+` | 684 | 0.1388 | [0.1295, 0.1488] |
| `2019_2022|W|0-5` | 9 | 0.2647 | [0.1460, 0.4312] |
| `2019_2022|W|6-11` | 19 | 0.5588 | [0.3945, 0.7112] |
| `2019_2022|W|12+` | 6 | 0.1765 | [0.0835, 0.3351] |
| `2023_2024|E|0-5` | 877 | 0.2354 | [0.2221, 0.2493] |
| `2023_2024|E|6-11` | 2348 | 0.6303 | [0.6147, 0.6457] |
| `2023_2024|E|12+` | 500 | 0.1342 | [0.1237, 0.1456] |
| `2023_2024|S|0-5` | 531 | 0.2245 | [0.2082, 0.2418] |
| `2023_2024|S|6-11` | 1529 | 0.6465 | [0.6270, 0.6655] |
| `2023_2024|S|12+` | 305 | 0.1290 | [0.1161, 0.1431] |
| `2023_2024|W|0-5` | 6 | 0.1818 | [0.0861, 0.3439] |
| `2023_2024|W|6-11` | 23 | 0.6970 | [0.5266, 0.8262] |
| `2023_2024|W|12+` | 4 | 0.1212 | [0.0482, 0.2733] |
| `2025|E|0-5` | 435 | 0.2292 | [0.2108, 0.2486] |
| `2025|E|6-11` | 1217 | 0.6412 | [0.6194, 0.6625] |
| `2025|E|12+` | 246 | 0.1296 | [0.1152, 0.1455] |
| `2025|S|0-5` | 268 | 0.2168 | [0.1948, 0.2407] |
| `2025|S|6-11` | 791 | 0.6400 | [0.6128, 0.6663] |
| `2025|S|12+` | 177 | 0.1432 | [0.1248, 0.1638] |
| `2025|W|0-5` | 3 | 0.2500 | [0.0889, 0.5323] |
| `2025|W|6-11` | 7 | 0.5833 | [0.3195, 0.8067] |
| `2025|W|12+` | 2 | 0.1667 | [0.0470, 0.4480] |
| `2026|E|0-5` | 420 | 0.2386 | [0.2193, 0.2591] |
| `2026|E|6-11` | 1133 | 0.6438 | [0.6211, 0.6658] |
| `2026|E|12+` | 207 | 0.1176 | [0.1034, 0.1335] |
| `2026|S|0-5` | 278 | 0.2219 | [0.1997, 0.2457] |
| `2026|S|6-11` | 800 | 0.6385 | [0.6115, 0.6646] |
| `2026|S|12+` | 175 | 0.1397 | [0.1216, 0.1600] |
| `2026|W|0-5` | 3 | 0.1667 | [0.0584, 0.3922] |
| `2026|W|6-11` | 14 | 0.7778 | [0.5478, 0.9100] |
| `2026|W|12+` | 1 | 0.0556 | [0.0099, 0.2576] |

## Confounders (not adjusted)

- platform: the inventory mixes Tenhou archive vintages (`tenhou_old`, `tenhou_recent`); client/rule changes and archive selection are not adjusted.
- rules: per-log rule flags (aka dora, hanchan vs tonpu, renchan/tobi, illegal action handling) are absent from the inventory and were NOT independently verified for every log, so rule drift is a candidate explanation, not a controlled variable.
- player pool: player identity is unavailable to this analysis; account rank, table level, time of day, region and repeated opponents are unknown and are not conditioned on.
- year: windows are fixed calendar partitions; seasonality, archive coverage, client updates and the January-only 2026 slice are not corrected for.
- AI provenance: all players are `unknown`. There is no local confirmed-AI evidence and no account list is published; absence of evidence is NOT `confirmed_human`.
- selection: the cohort is a deterministic hash-ranked per-year sample, not the full corpus; confidence intervals reflect sampled games only, not the data-generating uncertainty of all Tenhou play.
- counting: a ron event is a deal-in (multi-ron counts each winner); tsumo `hora` events carry `target == actor` in this corpus and are excluded from deal-in counts.

## Interpretation rules and explicit warning

- **`recent = stronger` is NOT a conclusion of this report.** The observed frequency changes are consistent with rules, player-pool, archive-selection and coverage confounders, all of which are unadjusted here.
- These are **frequency differences**, not demonstrated skill differences. Any quality claim needs conditioning, a pre-registered hypothesis and out-of-sample confirmation.
- 2026 is January-only locally; no annual claim is made for 2026.
- This run is explicitly filtered to years `[2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026]` (the span of the pre-registered windows); inventory rows outside that range were not sampled and no failure is implied by their absence.
- The sample is deterministic but partial; CIs reflect sampled games, not the whole corpus or the data-generating process.
- Multi-ron counts each winner; tsumo is not a deal-in.
- 2025/2026 zip members were plaintext and extracted copies are gzip, so parsing used compression magic sniffing rather than file extensions.
- No player identifiers or account lists are produced; optional HMAC aggregation is in-memory only.

## Teacher / AI provenance

- **Teacher audit: NOT_RUN.** No authorised teacher-evaluation protocol/config exists; local Mortal checkpoints have unverified distillation rights and were not used here.
- **AI provenance: all players `unknown`.** No local confirmed-AI evidence and no account list is published. Absence of evidence is not `confirmed_human`.

## Artefacts and reproduction

| file | rows | sha256 |
|---|---:|---|
| `drift_cohorts.csv` | 97 | `ef81de28e7d47923ca4e5422cd3612905d4af8bcecf2f83b5175f8240af1f138` |
| `drift_strata.csv` | 64 | `3b68e3a49033c0bd55df92e89866e285f14c9d21a4fc5c2f4e1d4c6ffbb8e80f` |

To reproduce, run the exact command above from the working directory after ensuring the recorded `libriichi` module path is importable. The per-year hash-ranked sample is fixed by the seed and manifest hash.
