# Analysis

Every number and figure of the paper and its Supplementary Information is computed here from the data deposit
(`--data DIR`, layout in `data/README.md`). Results go to `--out DIR` (default `<data>/analysis`). Nothing here calls a
model.

```bash
python3 -m analysis.run_all --data /path/to/deposit --figures              # statistics, numbers.tex, si_numbers.tex, figures
python3 -m analysis.run_all --data /path/to/deposit --replays --workers 8  # also redo the re-simulation stage (about 2 h)
```

The deposit already holds the outputs of the re-simulation stage (below), so the default run takes minutes.

## Order

**1. Re-simulation stage** (`analysis.replay_all`; slow). Every stored episode of a study is rebuilt and run again from its
cached model answers (no model is called; a missing answer stops the episode) and must equal its stored record on the breach,
breach step, breaching intruder, every intruder's outcome, the number of intruders destroyed and the steps run. One pass per study
takes all the measures that need a finished simulation:

| Output | Measure | Module |
|---|---|---|
| `coverage_exposure_<study>.json` | known / unseen / unpursued intruder time; attack exposure | `coverage_replay` |
| `main_replays.json` | connectivity, time-limit endings, safety-filter measures (main study) | `main_replays` |
| `behaviour_replays.json` | role counts before and after the first contact | `behaviour_replays` |
| `commander_commitments.json`, `commander_missed_orders.json` | duplicate commitments, orders that never arrived | `commander_replays` |
| `endings_<group>.json` | time-limit endings of the other studies | `endings_replays` |

Each module also runs alone (`python3 -m analysis.coverage_replay ...`).

**2. Statistics** (seconds to minutes; `python3 -m analysis.<module> --data DATA --out OUT`):

| Module | Output | What |
|---|---|---|
| `matrix_stats` | `matrix_analysis.json` | seed-clustered contrasts of the main study: t and bootstrap intervals, sign-flip p, Holm, Newcombe, McNemar |
| `high_authority` | `high_authority.json`, `high_authority_unseen.json` | consensus team at high authority against low authority |
| `high_authority_near_frontier` | `high_authority_near_frontier.json`, `..._secondary.json` | the same with the near-frontier model |
| `airspace` | `airspace.json`, `disguise.json` | birds and civilian drones; disguised intruders |
| `summary` | `summary.json` | near-frontier commander contrasts; coverage and exposure summaries |
| `key_rows` | `key_rows.json` | the key rows of the paper (K1, K2, K4, K5, X12, X13) |
| `run_facts` | `run_facts.json` | validity, model calls, retries, commitment conflicts |
| `behaviour_pool`, `replay_summary`, `lb_minus_li`, `bid_weights` | `*.json` | pooled behaviour counts, connectivity and filter summaries, burst minus independent loss, the bid weights the models wrote |
| `station_geometry`, `barrier_tests` | `*.json` | the simulator's station spacing; numerical tests of the barrier definition |

**3. Numbers** (`make_numbers`, `make_si_numbers`, `make_si_data`): `numbers.tex` (the macros the paper's text quotes),
`si_numbers.tex` and `si_tables/*.tex` (the Supplementary Information's macros and table rows), `Supplementary_Data_1.csv`. Each
macro carries the output leaf it was read from in a comment. The macro names are the identifiers of the paper's LaTeX source.

**4. Figures** (`figures/`, see `figures/README.md`).

## Checking

The deposit's `analysis/expected/` holds the outputs the paper was written from. Compare a fresh run with them:

```bash
diff <(sort OUT/numbers.tex | sed 's/ %.*//') <(sort expected/numbers.tex | sed 's/ %.*//')
```

Rerunning the chain reproduces every macro of `numbers.tex` and of `si_numbers.tex` (1,007), every row of the Supplementary
tables and `Supplementary_Data_1.csv`, and draws the figures of the paper pixel for pixel. The one difference from the paper's own
`numbers.tex` is a single unused macro that counted the records of runs that are not part of this repository, so the file holds
365 of its 366 macros.

## Seeds and conventions

Seeds 4001-4030 are the study; every contrast is first reduced to one number per seed. Percentages and percentage points are
rounded half up. `REF`, `MH`, `LI`, `LB` and `REF+attack` are the ideal network, limited range, independent loss, burst loss
and falsified reports. `cbaa`, `independent`, `commander` and `cbaa_sw2` are the consensus team at low authority, the
independent team, the commander team and the consensus team at high authority (README, "Names in the code and in the paper").
