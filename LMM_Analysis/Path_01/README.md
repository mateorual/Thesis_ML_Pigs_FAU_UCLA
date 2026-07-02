# LMM Analysis - Path 01

Path 01 performs treatment-specific longitudinal mixed-effects modeling for the thesis acoustic analysis of Yucatan mini-pig squeals.

This path uses the selected-feature dataset formed from the union of features selected in the scar-only (`S`), unilateral COVR (`U`), and bilateral COVR (`B`) groups. A separate linear mixed-effects model is fitted for each treatment group and each selected feature.

## Goal

Path 01 answers the within-treatment longitudinal question:

> For each treatment group, which selected acoustic features deviate from the pre-surgical baseline after surgery, and do these deviations appear temporary or persistent?

The analysis fits group-specific LMMs and then classifies features according to their effect-size pattern across recovery stages.

## Repository Contents

```text
Path_01/
├── data/
│   └── Selected_Features_Groups_S_U_B_v1.xlsx
├── scripts/
│   ├── run_group_level_lmm_effect_sizes.py
│   └── classify_temporary_permanent_features.py
├── outputs/
│   └── reference_results/
│       ├── effect_size_LMM_results_group_level.xlsx
│       ├── temporary_permanent_group_level.xlsx
│       └── lmm_analysis_warnings.log
├── requirements.txt
└── README.md
```

Generated outputs from a new run are written to:

```text
outputs/generated/
```

The included `outputs/reference_results/` files are the thesis results generated from the original analysis run.

## Input Data

Main input:

```text
data/Selected_Features_Groups_S_U_B_v1.xlsx
```

Expected sheet:

```text
Features_Data
```

Expected metadata columns:

```text
Squeal, Subject, Treatment, Week, Num_Cycles, Stage
```

All remaining columns are selected acoustic features. The included input has:

- 2,356 squeals
- 6 metadata columns
- 59 selected acoustic features

The analysis uses the stages:

```text
Pre
Early Post
Late Post
```

`Mid Post` is dropped in this Path 01 analysis.

## Model

For each treatment group and feature, the model is:

```text
feature ~ Stage + (1 | Subject)
```

where:

- `feature` is z-scored before model fitting
- `Stage` is categorical with `Pre-Surgery` as the reference level
- `Subject` is treated as a random intercept

The script estimates three contrasts:

```text
Pre vs Early Post
Pre vs Late Post
Early Post vs Late Post
```

Because the outcome feature is standardized before fitting, the fixed-effect coefficient is interpreted as Cohen's d.

## Treatment Groups

The analysis is run independently for:

| Code | Treatment group | Subject units |
|------|-----------------|---------------|
| `S` | Scar-only | Barry and Stevie; Cher and Adele; Elvis; Snoop and Dre |
| `U` | Unilateral COVR | Michael and Prince; Paul and John; Tina and Aretha |
| `B` | Bilateral COVR | Beck and Kurt; Frank and Dean; Madonna and Beyonce; Taylor and Gaga |

## Outputs

### 1. Effect-size LMM workbook

Produced by:

```text
scripts/run_group_level_lmm_effect_sizes.py
```

Output:

```text
outputs/generated/effect_size_LMM_results_group_level.xlsx
```

The workbook contains:

- `LMM_B_PreVsEarly`
- `LMM_B_PreVsLate`
- `LMM_B_EarlyVsLate`
- `LMM_U_PreVsEarly`
- `LMM_U_PreVsLate`
- `LMM_U_EarlyVsLate`
- `LMM_S_PreVsEarly`
- `LMM_S_PreVsLate`
- `LMM_S_EarlyVsLate`
- `Summary`

Each contrast sheet includes:

- beta estimate
- standard error
- Kenward-Roger degrees of freedom when available through `pymer4`
- t-statistic
- raw and FDR-adjusted p-values
- Cohen's d
- effect-size interpretation
- confidence interval
- marginal and conditional R-squared
- convergence and low-power flags

### 2. Temporary/permanent classification workbook

Produced by:

```text
scripts/classify_temporary_permanent_features.py
```

Output:

```text
outputs/generated/temporary_permanent_group_level.xlsx
```

The classification uses the effect-size interpretation from the key contrasts:

- **Temporary:** large Pre vs Early effect, but negligible/small Pre vs Late effect
- **Permanent:** large Pre vs Early effect and large Pre vs Late effect
- **Other:** all remaining patterns or non-converged model combinations

The workbook contains:

- `GroupLevel_B`
- `GroupLevel_U`
- `GroupLevel_S`
- `Summary`
- `CrossGroup_Consensus`

## Installation

Create and activate a Python environment:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

On macOS/Linux:

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

## R Dependency

This path uses `pymer4`, which requires R and R mixed-model packages. On Windows, the script first uses an existing `R_HOME` value and otherwise attempts to detect the newest R installation under `C:\Program Files\R`. If your R installation is elsewhere, set `R_HOME` before running the script.

Required R-side packages typically include:

```r
install.packages(c("lme4", "lmerTest", "pbkrtest"))
```

## Run Path 01

From this folder:

```bash
python scripts\run_group_level_lmm_effect_sizes.py
python scripts\classify_temporary_permanent_features.py
```

The first script can take time because it fits one LMM per selected feature and treatment group.

The classification script reads:

1. `outputs/generated/effect_size_LMM_results_group_level.xlsx` if it exists, otherwise
2. `outputs/reference_results/effect_size_LMM_results_group_level.xlsx`

This means the classification step can be tested directly using the included reference workbook.

## Notes

- Path 01 is a within-treatment analysis. It does not directly test S vs U treatment differences.
- Direct S vs U trajectory comparison is handled in Path 02.
- The `U` group has three pig-pair subject units, so p-values should be interpreted with caution. The effect-size trajectories are the main descriptive signal for this group.
- The included reference workbooks correspond to the thesis results generated from this implementation.
