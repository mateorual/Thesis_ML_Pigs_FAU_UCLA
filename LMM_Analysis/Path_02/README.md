# LMM Analysis — Path 02: Joint Scar-Only (S) vs Unilateral COVR (U) Model

Reproducible R pipeline (with one supporting Python figure script) that tests whether the
post-surgical acoustic recovery trajectory of pig squeals differs between the **scar-only (S)**
and **unilateral COVR (U)** treatment groups, using linear mixed-effects models (LMMs) on 50
acoustic features selected specifically for the S-vs-U comparison.

## How this fits into the thesis

This is **Path_02** of the `LMM_Analysis` component. The thesis's longitudinal modeling used two
complementary designs:

- **Path_01** (`LMM_Analysis/Path_01`) fits a separate model *within* each treatment group (S, B,
  U analyzed independently) using the feature set selected from the S+U+B union, and classifies
  each feature's recovery pattern (temporary vs. permanent) from the fitted effect sizes.
- **Path_02** (this folder) instead fits **one joint model across S and U together**, with
  `Stage * Treatment` as the fixed-effect structure, so that the `Stage:Treatment` interaction
  term directly tests whether the two groups' trajectories differ — a more direct statistical
  comparison than looking at the two per-group models from Path_01 side by side. It uses a
  separate, smaller feature set selected specifically from the S+U union (50 features), not the
  S+U+B union used by Path_01.

Both groups involve unilateral vocal-fold injury; only U additionally received the COVR implant.
The S-vs-U comparison therefore isolates the effect of the implant from the effect of the injury
itself.

## Project Structure

```
Path_02/
├── README.md
├── Path_02.Rproj                 # open this in RStudio to set the working directory automatically
├── requirements.txt               # Python deps for the one supporting figure script
├── config/
│   └── analysis_config.yml        # reference documentation of the pipeline's parameters
├── docs/
│   └── methodology_notes.md       # statistical rationale for each design decision
├── scripts/
│   ├── 00_setup.R                 # paths, package loading, shared helpers
│   ├── 01_data_audit.R            # load data, validate structure, missingness, sample-size counts
│   ├── 02_lmm_primary_analysis.R  # fits feature ~ Stage*Treatment + (1|Subject) per feature
│   ├── 03_nonparametric_secondary_analysis.R  # Kruskal-Wallis + Wilcoxon (descriptive only)
│   ├── 04_reporting_tables.R      # manuscript-ready tables, draft methods paragraph
│   ├── 05_diagnostic_plots.R      # per-feature residual diagnostics
│   ├── 06_trajectory_plots.R      # per-feature S vs U trajectory plots (raw + LS means)
│   ├── 07_supplement_outputs.R    # BH/BY-adjusted p-values, AIC/BIC, SE/CI, extra figures
│   ├── 08_spaghetti_plots.R       # individual pig-pair trajectory ("spaghetti") plots
│   ├── run_all.R                  # sources 00-08 in order
│   ├── run_key_panel.R            # standalone: 5-feature LS-mean summary panel (run after run_all.R)
│   └── figure_longitudinal_boxplots.py  # independent Python script; does not depend on the R pipeline
├── Selected_Features_Groups_S_U_v1.xlsx   # input data (see below)
├── box_plots/                     # output of figure_longitudinal_boxplots.py
├── logs/
│   └── pipeline.log                # timestamped log from the shipped run
└── results/                       # output of the R pipeline (see Output below)
```

## Statistical Model

```
feature ~ Stage * Treatment + (1 | Subject)
```

- **Stage** (fixed, unordered categorical, reference = `Pre`): `Pre`, `Early Post`, `Mid Post`,
  `Late Post`. Modeled as categorical rather than continuous week because recordings are sparse
  and irregularly spaced; a continuous model would interpolate across gaps that don't represent a
  smooth physiological trajectory.
- **Treatment** (fixed, reference = `S`): `S` vs `U`.
- **Stage × Treatment** — the primary inferential term. Tests whether the recovery trajectory
  differs between groups, not just whether either group changes from baseline.
- **Subject** (random intercept) — the pig pair (or individual, for unpaired subjects). Squeals
  from the same pig pair are correlated, so treating each squeal as an independent observation
  would understate uncertainty.
- Estimated by REML with Satterthwaite degrees-of-freedom approximation; fixed effects tested
  with Type III F-tests (appropriate for an unbalanced design with an interaction term).
- Omnibus p-values (Stage, Treatment, Stage:Treatment) are corrected for multiple comparisons
  across the 50 features with both Benjamini-Hochberg (BH, less conservative) and
  Benjamini-Yekutieli (BY, valid under arbitrary correlation between features — relevant since
  acoustic features are correlated).

Full rationale for each of these choices is in [`docs/methodology_notes.md`](docs/methodology_notes.md).

Kruskal-Wallis and Wilcoxon tests are also computed (`03_nonparametric_secondary_analysis.R`) but
are **secondary/descriptive only** — they treat squeals as independent observations, which
overstates significance given the repeated-measures structure. They are not used for inference.

## Input Data

| Field | Value |
|---|---|
| File | `Selected_Features_Groups_S_U_v1.xlsx` |
| Sheet | `Features_Data` |
| Rows | 1,486 squeal events |
| Features | 50 acoustic features (55 total columns = 50 features + 5 metadata columns: `Squeal`, `Pig_ID`, `Treatment`, `Week`, `Stage`) |
| Treatments | S — Barry and Stevie, Cher and Adele, Elvis, Snoop and Dre (4 pig pairs/individuals) · U — Michael and Prince, Paul and John, Tina and Aretha (3 pig pairs) |
| Stages | Pre, Early Post, Mid Post, Late Post |

`01_data_audit.R` renames the input's `Pig_ID` column to `Subject` and validates that exactly 50
feature columns are present (warns, but does not stop, if the count differs).

This file is the S-only-selected-features counterpart to
[`Feature_Selection/outputs/selected_feature_datasets/Selected_Features_Groups_S_U_v1.xlsx`](../../Feature_Selection/outputs/selected_feature_datasets/Selected_Features_Groups_S_U_v1.xlsx)
in this repository. The two are the same underlying feature-selection run (identical feature
values for every squeal present in both), but are not byte-identical snapshots — the copy here is
the exact input that produced the `results/` in this folder, so re-running the pipeline against it
reproduces the shipped outputs exactly.

## How to Run

### R pipeline (primary analysis)

Requires R 4.5.2 (or compatible) and the packages listed at the top of `scripts/00_setup.R`
(`readxl`, `writexl`, `openxlsx`, `dplyr`, `tidyr`, `purrr`, `stringr`, `forcats`, `lme4`,
`lmerTest`, `emmeans`, `broom.mixed`, `performance`, `DHARMa`, `ggplot2`, `ggtext`, `rstatix`,
`janitor`, plus `tibble` and, for the optional panel script, `patchwork` — all installed
automatically by `00_setup.R` if missing).

1. Open `Path_02.Rproj` in RStudio (this sets the working directory to `Path_02/` automatically),
   or manually `setwd("<path to>/Path_02")` in a plain R session.
2. Run:
   ```r
   source("scripts/run_all.R")
   ```
   This sources `00_setup.R` through `08_spaghetti_plots.R` in order and writes everything under
   `results/`. Each script can also be re-run individually afterward (they each check whether
   `00_setup.R`/`01_data_audit.R` have already been sourced in the session and, if not, source
   them first).
3. Optionally, after step 2 has populated `results/tables/`, generate the 5-feature summary panel:
   ```r
   source("scripts/run_key_panel.R")
   ```

### Python figure script (independent of the R pipeline)

```bash
pip install -r requirements.txt
python scripts/figure_longitudinal_boxplots.py
```

Reads `Selected_Features_Groups_S_U_v1.xlsx` directly and writes to `box_plots/`. Does not read
or depend on any R pipeline output.

## Output

### `results/` (from the R pipeline)

| Folder | Contents |
|---|---|
| `manuscript_ready/` | Final consolidated tables and draft methods text — start here |
| `tables/` | All intermediate and full-detail xlsx/csv tables (ANOVA, LS means, raw means, pairwise contrasts, diagnostics, audit counts) |
| `models/` | One serialized `lmer` model object per feature (`<feature>_lmm.rds`), so results can be re-inspected without refitting |
| `diagnostics/` | Two diagnostic plots per feature: a 4-panel residual/QQ/scale-location/leverage plot and a DHARMa simulated-residuals plot |
| `figures/Original/` | Per-feature trajectory plots (raw means ± SE), plus the subject×stage sample-size heatmap |
| `figures/LS_Mean/` | Per-feature LS-mean trajectory plots ± 95% CI, plus (if `run_key_panel.R` was run) the 5-feature summary panel |
| `figures/Spaghetti/` | Per-feature individual pig-pair trajectory plots, plus a combined key-features panel |
| `descriptive_pair_level/` | Pair-level Wilcoxon results (secondary/descriptive) |
| `logs/model_failures.csv` | Any features whose model fitting failed or fell back to the additive (non-interaction) formula |

Key files in `results/manuscript_ready/`:

| File | Contents |
|---|---|
| `Table_LMM_Main_Results.xlsx` | One row per feature: omnibus p-values (raw + BH + BY), AIC/BIC/logLik, variance components, singular-fit/convergence flags |
| `Table_Raw_and_LSMeans.xlsx` | Raw means ± SE/CI and LS means ± 95% CI per feature × Stage × Treatment |
| `Methods_Statistical_Analysis.txt` | Draft methods paragraph for the manuscript |

### `box_plots/` (from the Python script)

`longitudinal_boxplots_S.png/.pdf` and `longitudinal_boxplots_U.png/.pdf` (per-group distribution
boxplots with Wilcoxon significance brackets), plus `longitudinal_significance_BY.xlsx`.

## Notes

- **All 50 models fit successfully with the full `Stage * Treatment` interaction** — verified
  directly against the shipped `results/manuscript_ready/Table_LMM_Main_Results.xlsx`,
  `results/tables/lmm_model_status_log.xlsx`, and `results/tables/lmm_diagnostics_summary.xlsx`:
  zero singular fits, zero convergence warnings, zero fallbacks to the additive formula, across
  all three tables.
- `figure_longitudinal_boxplots_SUB.py` and `Selected_Features_Manually_S_U_B.xlsx` from the
  original analysis folder (a three-group S/U/B boxplot variant) are **not included here** —
  they're out of scope for Path_02, which is specifically the S-vs-U joint comparison.
- `config/analysis_config.yml` is a human-readable reference summary of the pipeline's parameters;
  it is not read programmatically by any script. The scripts under `scripts/` are the source of
  truth for actual behavior.
- Re-run from scratch by deleting `results/` and `logs/pipeline.log`, then re-running
  `scripts/run_all.R`.

## Troubleshooting

**Scripts can't find `00_setup.R` / `01_data_audit.R` / the input xlsx** — the R working directory
must be `Path_02/` itself. Open `Path_02.Rproj` in RStudio (recommended), or call
`setwd("<path to>/Path_02")` before sourcing anything.

**`WARNING: expected 50 feature columns, found <n>`** — printed by `01_data_audit.R` if the input
file's column layout doesn't match what the rest of the pipeline expects; check that
`Selected_Features_Groups_S_U_v1.xlsx` has exactly 5 metadata columns
(`Squeal`, `Subject`/`Pig_ID`, `Treatment`, `Week`, `Stage`) before the feature columns begin.

**`run_key_panel.R` fails to find `lmm_lsmeans_stage_treatment_all_features.xlsx`** — run
`scripts/run_all.R` (specifically `02_lmm_primary_analysis.R`) first to populate `results/tables/`.
