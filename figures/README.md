# Figures

Every figure of the paper and of the Supplementary Information is drawn from the analysis outputs (`analysis/README.md`) and
the data deposit. Run each as a module after `analysis.run_all`:

```bash
python3 -m figures.fig_contrasts --data DATA [--out OUT] [--figures DIR]
```

`--out` is the analysis results directory (default `<data>/analysis`); the figures go to `--figures` (default `<out>/figures`).
Each script asserts that every number it prints or plots equals the macro in `numbers.tex` / `si_numbers.tex` and that the
plotted per-seed values reproduce the stored estimate and interval, so a figure cannot drift from the text. `style.py` is the
one place that sets colours, fonts, sizes and strokes.

| Figure | Script | Needs |
|---|---|---|
| 1 how the teams decide | `fig_architecture` | `assets/contact_photos/intruder_1.jpeg` |
| 2 the arena | `fig_arena` | the simulator's geometry |
| 3 breach rates and team contrasts | `fig_contrasts` | `matrix_analysis.json`, records |
| 4 allocation | `fig_allocation` (+ `allocation_data`) | the re-simulation outputs |
| 5 limited range | `fig_range` (+ `range_data`) | the re-simulation outputs |
| 6 authority and model capability | `fig_authority` | high-authority outputs |
| 7 airspace challenges | `fig_airspace` | the bundle `fig7a_cbaa` (`python3 -m analysis.episode_bundles --data DATA fig7a_cbaa`), Google Chrome, and the four bird-drone photographs (`--bird-drone-dir`, not part of this repository) |
| Supplementary breach grid | `fig_breach_grid` | records |
| Supplementary safety-filter figure | `fig_filter_headon` | `barrier_tests.json` |
| Supplementary episode figures | `episodes_si` | the bundles of `python3 -m analysis.episode_bundles --data DATA`, Google Chrome |

The episode figures, the videos and the README GIFs are all drawn by the replay viewer (`viewer/`) from the same bundles with
`scripts/replay_media.py`. `tests/test_viewer_tokens.py` pins the viewer's colours and sizes to `figures/style.py`.
