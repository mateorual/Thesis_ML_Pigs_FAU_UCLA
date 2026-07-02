# Methodology Notes — S vs U LMM Pipeline

## Dataset
- 1,486 squeal events from 7 pig pairs (4 S-group, 3 U-group)
- 50 acoustic features selected from a larger feature set
- Recordings span weeks 0–28; time is collapsed into 4 categorical stages

## Why categorical Stage (not continuous Week)
Recordings are sparse and irregular across pigs and weeks. Using `Week` as a continuous
predictor would interpolate between gaps that do not represent a smooth physiological
trajectory. Categorical `Stage` maps to the clinically defined perioperative periods
(Pre-operative, Early Post, Mid Post, Late Post) and avoids assumptions about the
shape of the time-course.

## Why mixed effects (LMM) as primary analysis
Each pig pair contributes many squeals, and squeals from the same pair are correlated.
Ignoring this correlation inflates degrees of freedom and produces anti-conservative
p-values. The random intercept for `Subject` (pig pair) accounts for between-pair
baseline differences, so fixed-effect tests reflect within-pair changes.

## Why LS means / EMMs in addition to raw means
The data are unbalanced: different pig pairs contribute different numbers of squeals at
different stages, partly because of recording gaps and pig availability. Raw means
conflate the group-level effect with the composition of the sample. LS means estimated
from the model hold the random effect at its average and produce cell estimates that
are directly comparable across Treatment × Stage cells.

## Type III tests
Type III tests evaluate each fixed effect after accounting for all others. This is the
appropriate choice when the design is unbalanced and when interaction terms are present,
because it tests the marginal effect of each term. The reference level for `Treatment`
is `S`; for `Stage`, the factor is unordered so all pairwise stage contrasts are
estimable symmetrically.

## Multiple testing across 50 features
Omnibus p-values (Stage, Treatment, Stage × Treatment) are reported with:
- **BH (Benjamini-Hochberg)**: controls the false discovery rate, appropriate for
  exploratory science where some true effects are expected.
- **BY (Benjamini-Yekutieli)**: more conservative; valid under arbitrary correlation
  between tests, which may be present since acoustic features are correlated.

The interaction term `Stage × Treatment` is the primary inferential focus.

## Pairwise post-hoc contrasts
Treatment-within-Stage and Stage-within-Treatment contrasts are computed from the
model-based marginal means (emmeans). BH adjustment is applied within each contrast
family. These are labeled *exploratory post-hoc* because they were not pre-specified
and are secondary to the omnibus interaction test.

## Wilcoxon / Kruskal-Wallis — secondary descriptive only
These nonparametric tests treat observations (squeals) as independent, which they are
not: squeals from the same pig pair at the same stage are correlated. Using them for
primary inference would understate uncertainty. They are retained as a descriptive
sanity check only and labeled accordingly in all outputs.

## Singular fits
Seven models returned a singular fit warning (random intercept variance at the
boundary of zero). This indicates that for those features, the pig-pair level
variation is negligible relative to within-pair variation. Results are still valid
but should be interpreted noting that the random effect provides little correction.

## Software
- R 4.5.2
- lme4, lmerTest (LMM fitting)
- emmeans (LS means and contrasts)
- writexl, openxlsx (Excel outputs)
- ggplot2 (figures)
