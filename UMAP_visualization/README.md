# UMAP Visualization

UMAP (Uniform Manifold Approximation and Projection) dimensionality-reduction plots of pig squeal
acoustic features, used as an exploratory check of whether squeals from different perioperative
stages occupy separable regions of the acoustic feature space, within each treatment group and
across all groups combined.

> The reference implementation this repo is based on was named
> `Visualizations_tSNE_UMAP_PCA_June10`, but no t-SNE or PCA code actually exists in it — only the
> project folder name mentions them. This repo is UMAP only.

These plots are **qualitative/exploratory checks, not inferential evidence**: they are used to see
whether apparent clustering is driven by a single pig or is visible across multiple subjects, not
to test a hypothesis. The statistical inference on treatment/stage effects is done separately by
the `LMM_Analysis` component.

## Project Structure

```
UMAP_visualization/
├── README.md
├── requirements.txt
├── scripts/
│   ├── plot_umap_groups.py                        # per-treatment group UMAP sweep (full features)
│   ├── plot_umap_groups_selected_features.py       # same, on the selected-feature dataset
│   ├── plot_umap_all_treatments.py                 # all treatments combined, one embedding each
│   ├── plot_umap_individual_pigs.py                # per-pig UMAP sweep (full features)
│   ├── plot_umap_individual_pigs_selected_features.py  # same, on the selected-feature dataset
│   └── plot_combined_thesis_figures.py             # stitches curated panels into the final figure
├── Consolidated_Features_Groups_S_U_B_v1.xlsx       # input: full feature matrix (474 features)
├── Selected_Features_Groups_S_U_B_v1.xlsx           # input: selected feature matrix (58 features)
├── umap_plots/                                      # output of the *_groups/_individual_pigs/_all_treatments scripts, full features
│   ├── All_Treatments/                              # all 3 groups combined, one embedding
│   ├── Bilateral/                                   # group embedding + Pig 2&3/4&5/6&7/12&13 subfolders + ThesisFigure/
│   ├── Scar/                                        # group embedding + Pig 1/8&9/16&17/18&19 subfolders + ThesisFigure/
│   └── Unilateral/                                  # group embedding + Pig 10&11/14&15/20&21 subfolders + ThesisFigure/
└── umap_plots_selected_features/                    # same structure, on the selected-feature dataset (no All_Treatments variant)
    ├── Bilateral/
    ├── Scar/
    └── Unilateral/
```

## Input Data

| File | Rows | Columns | Features | Notes |
|---|---|---|---|---|
| `Consolidated_Features_Groups_S_U_B_v1.xlsx` | 2,356 squeals | 481 | 474 | Full acoustic feature matrix across all three treatment groups, enriched with a categorical `Stage` label (derived from week relative to surgery) |
| `Selected_Features_Groups_S_U_B_v1.xlsx` | 2,356 squeals | 65 | 58 | The S+U+B selected-feature subset (same feature-selection output used by `LMM_Analysis/Path_01`), similarly enriched with `Stage` |

Both files share the same 7 metadata columns: `Squeal`, `Subject`, `Pig_ID`, `Treatment`
(`B`/`S`/`U`), `Week`, `Num_Cycles`, `Stage` (`Pre`, `Early Post`, `Mid Post`, `Late Post`).

Treatment groups and their pigs: **B** (Bilateral COVR) — Pig 2&3, 4&5, 6&7, 12&13 · **S**
(Scar-only) — Pig 1, 8&9, 16&17, 18&19 · **U** (Unilateral COVR) — Pig 10&11, 14&15, 20&21.

`Selected_Features_Groups_S_U_B_v1.xlsx` is the same underlying selection as
[`Feature_Selection/outputs/selected_feature_datasets/Selected_Features_Groups_S_U_B_v1.xlsx`](../Feature_Selection/outputs/selected_feature_datasets/Selected_Features_Groups_S_U_B_v1.xlsx)
elsewhere in this repository, but is not a byte-identical snapshot (as with `LMM_Analysis/Path_02`,
each component keeps the exact snapshot that produced its own shipped outputs, so results stay
reproducible from the files checked in alongside them).

## How the Figures Were Produced

Producing the final thesis figures is a **three-stage process**, and only the first and third
stages are scripted:

1. **Hyperparameter sweep** (scripted). `plot_umap_groups.py`, `plot_umap_groups_selected_features.py`,
   `plot_umap_all_treatments.py`, and the two `plot_umap_individual_pigs*.py` scripts each fit UMAP
   at several `(n_neighbors, min_dist)` configurations — group-level scripts use 5–7 configs, per-pig
   scripts use 8 — and save every resulting embedding as a PNG under `umap_plots/` or
   `umap_plots_selected_features/`. This produces the bulk of the PNGs in this repo (217 + 147 = 364
   files) and is retained for transparency into what was explored, not because every configuration
   is individually meaningful.
2. **Manual curation** (not scripted). For each treatment group, one group-level config and one
   config per pig were judged to best show the group/individual structure, and those specific PNGs
   were copied into a `ThesisFigure/` subfolder. **This repository ships the already-curated
   `ThesisFigure/` folders**, so this step does not need to be repeated to reproduce the final
   figures from what's checked in — it only matters if you regenerate the sweep from different data
   and need to choose new panels.
3. **Combine into the final figure** (scripted). `plot_combined_thesis_figures.py` finds every
   `ThesisFigure/` folder under the project root, takes whatever group plot and pig plots are
   sitting in it, and stitches them into a single multi-panel figure: Panel A (large, left) is the
   group embedding, Panels B onward (stacked, right) are the individual-pig embeddings. Output is
   `combined_figure_all_{B,S,U}.png/.pdf` (from `umap_plots/`) and
   `combined_figure_selected_{B,S,U}.png/.pdf` (from `umap_plots_selected_features/`).

**`combined_figure_all_{B,S,U}.pdf` are the actual figures used in the thesis** (Methodology
chapter, full-feature-space UMAP panels).

## How to Run

```bash
pip install -r requirements.txt
```

All scripts resolve their own paths relative to their location, so they can be run from anywhere,
e.g. from the repository root:

```bash
python scripts/plot_umap_groups.py
python scripts/plot_umap_groups_selected_features.py
python scripts/plot_umap_all_treatments.py
python scripts/plot_umap_individual_pigs.py
python scripts/plot_umap_individual_pigs_selected_features.py
```

Each of these five scripts creates its own output folders as needed and can be re-run
independently and in any order; re-running overwrites only the PNGs it produces.

Then, once the sweep outputs exist and `ThesisFigure/` folders are populated (already the case for
the folders shipped in this repo):

```bash
python scripts/plot_combined_thesis_figures.py
```

This scans the whole project for `ThesisFigure/` folders and regenerates every combined figure it
finds.

## Output

| Folder | From | Contents |
|---|---|---|
| `umap_plots/All_Treatments/` | `plot_umap_all_treatments.py` | All 3 treatment groups in one embedding, colored by treatment, marker = stage; 3-stage and 4-stage variants × 5 configs |
| `umap_plots/{Bilateral,Scar,Unilateral}/` | `plot_umap_groups.py` | Group-level embedding, colored by stage, marker = Pig ID; 3-stage and 4-stage variants × 7 configs |
| `umap_plots/{Bilateral,Scar,Unilateral}/Pig */` | `plot_umap_individual_pigs.py` | Per-pig embedding (3-stage only), colored by stage; 8 configs |
| `umap_plots/{Bilateral,Scar,Unilateral}/ThesisFigure/` | curated + `plot_combined_thesis_figures.py` | The one group plot + per-pig plots chosen for the manuscript, plus the combined multi-panel figure |
| `umap_plots_selected_features/...` | same scripts, `*_selected_features.py` variants | Same structure, run on the 58-feature selected dataset (no `All_Treatments` variant) |

Filenames encode the UMAP hyperparameters directly, e.g.
`umap_bilateral_3stages_cfg3_nn15_md0p3.png` = Bilateral group, 3-stage variant, config 3,
`n_neighbors=15`, `min_dist=0.3`.

## Notes

- All five sweep/combine scripts standardize features first: missing values are mean-imputed
  (`sklearn.impute.SimpleImputer`), then features are z-scored (`sklearn.preprocessing.StandardScaler`)
  before UMAP fitting. `random_state=42` is fixed for reproducibility.
- `plot_umap_individual_pigs.py` (the full-feature-set version) originally **silently skipped any
  pig whose output subfolder did not already exist**, rather than creating it — meaning a fresh
  clone of the original folder without those pre-existing empty subfolders would have produced no
  per-pig plots at all with no error, only a "folder not found" message per pig. Fixed here to
  create the folder like every other script in this repository does (and like its own
  `_selected_features` counterpart already did correctly).
- `umap_plots_selected_features/` has no `All_Treatments/` folder — there is no
  `plot_umap_all_treatments_selected_features.py` script; the all-treatments combined view was only
  produced for the full feature set.

## Troubleshooting

**`plot_combined_thesis_figures.py` prints `WARNING: no group plot found` / `no pig plots found`**
— the corresponding `ThesisFigure/` folder is missing its curated group or per-pig PNG(s). This is
a manual curation step (see [How the Figures Were Produced](#how-the-figures-were-produced)); copy
the desired sweep-output PNG(s) into that `ThesisFigure/` folder before re-running.

**A script fits UMAP but produces near-empty or garbled plots** — check that the input file's
`Treatment`/`Stage` values match the expected sets (`B`/`S`/`U` and `Pre`/`Early Post`/`Mid
Post`/`Late Post`); the scripts silently produce empty subsets (zero-point scatter) rather than
erroring if a filter matches nothing.
