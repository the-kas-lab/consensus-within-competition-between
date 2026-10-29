# Data directory

Put the data deposit here, or anywhere else and pass its path as `--data`. Nothing in this folder except this
file is part of the repository.

Layout (`experiments/studies.py` defines it):

```
<data>/<study>/<run>/records.jsonl                 one record per episode: configuration, outcome, counters
<data>/<study>/<run>/records.jsonl.identity.json   the execution identity of the run
<data>/<study>/<run>/cache/<identity>/<episode>/   every model answer of the episode, stored under a hash of its request
<data>/<study>/<run>/cache/<identity>/<episode>__near_frontier/   the near-frontier model's answers, where it served a seat
```

Besides the six studies the deposit holds:

```
<data>/disguise_probe/calls.jsonl     the 192 single-photograph calls of the photo probe (model, photograph, framing, answer)
<data>/analysis/                      the results of the analysis scripts (analysis/README.md), including the outputs of the
                                      re-simulation stage and the nine replay bundles in analysis/bundles/
<data>/analysis/expected/             the paper's own numbers.tex, si_numbers.tex (names and values), Supplementary_Data_1.csv
                                      and the Supplementary tables' rows, to compare a fresh run with
```

| Study | Runs | Episodes |
|---|---|---|
| `main_study` | `seeds_4001_4010`, `seeds_4011_4030` | 900 (3 teams x 2 intruder counts x 5 network conditions x 30 seeds) |
| `high_authority_low_cost` | `seeds_4001_4030` | 300 |
| `high_authority_near_frontier` | `first_run`, `second_run` | 28 + 8 (the second run repeats the 8 seeds whose first episodes failed under a provider's rate limit) |
| `near_frontier_commander` | `seeds_4001_4010`, `seeds_4011_4030` | 40 + 80 |
| `airspace` | `all_arms` | 25 (5 teams x 5 seeds) |
| `disguise` | `all_arms` | 50 (5 teams x 5 seeds x without and with the forewarning) |

Replay a run with `python3 -m experiments.run <study> --data <data> --replay`.
