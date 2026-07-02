# 04_reporting_tables.R — Manuscript-ready summary tables and methods draft

`%+%` <- function(a, b) paste0(a, b)

if (!exists("dat_clean")) {
  source("scripts/00_setup.R")
  source("scripts/01_data_audit.R")
}

# Load LMM results if not already in memory (or if NULL from pipeline)
load_if_needed <- function(var_name, path) {
  if (!exists(var_name, envir = .GlobalEnv) || is.null(get(var_name, envir = .GlobalEnv))) {
    assign(var_name, readxl::read_excel(path), envir = .GlobalEnv)
  }
}
load_if_needed("lmm_anova_all",   file.path(dir_tables, "lmm_type3_anova_all_features.xlsx"))
load_if_needed("lmm_lsmeans_all", file.path(dir_tables, "lmm_lsmeans_stage_treatment_all_features.xlsx"))
load_if_needed("lmm_raw_all",     file.path(dir_tables, "raw_means_by_stage_treatment_all_features.xlsx"))
load_if_needed("lmm_status_all",  file.path(dir_tables, "lmm_model_status_log.xlsx"))

# ── Helper: pivot raw means to wide (one column per Treatment × Stage) ─────────
pivot_raw_wide <- function(raw_df) {
  raw_df %>%
    dplyr::select(feature, Treatment, Stage, mean, sd, n_squeals, n_subjects) %>%
    dplyr::mutate(
      col_mean = "raw_mean_" %+% Treatment %+% "_" %+% gsub(" ", "_", Stage),
      col_sd   = "raw_sd_"   %+% Treatment %+% "_" %+% gsub(" ", "_", Stage)
    ) %>%
    tidyr::pivot_wider(
      id_cols    = feature,
      names_from = c(Treatment, Stage),
      values_from = mean,
      names_glue = "raw_mean_{Treatment}_{gsub(' ', '_', Stage)}"
    )
}

# Wide raw means
raw_wide <- tryCatch({
  lmm_raw_all %>%
    dplyr::mutate(col = "raw_mean_" %+% Treatment %+% "_" %+% gsub(" ", "_", Stage)) %>%
    dplyr::select(feature, col, mean) %>%
    tidyr::pivot_wider(names_from = col, values_from = mean)
}, error = function(e) NULL)

# ── 1. Main LMM summary table ─────────────────────────────────────────────────
# Extract interaction p-value from anova table
pval_cols <- c("Pr(>F)", "p.value")
get_pval <- function(anova_df, term_pattern) {
  tryCatch({
    row_idx <- grep(term_pattern, anova_df$term, ignore.case = TRUE)
    pcol    <- intersect(pval_cols, names(anova_df))[1]
    if (length(row_idx) == 0 || is.na(pcol)) return(NA_real_)
    anova_df[[pcol]][row_idx[1]]
  }, error = function(e) NA_real_)
}

# Identify p-value column (lmerTest anova uses "Pr(>F)")
pcol_anova <- intersect(pval_cols, names(lmm_anova_all))[1]
if (is.na(pcol_anova)) pcol_anova <- names(lmm_anova_all)[ncol(lmm_anova_all)]

extract_p <- function(sub_df, term_pattern) {
  row <- grep(term_pattern, sub_df$term, ignore.case = TRUE)
  if (length(row) == 0) return(NA_real_)
  sub_df[[pcol_anova]][row[1]]
}

anova_pvals <- lmm_anova_all %>%
  dplyr::group_by(feature) %>%
  dplyr::summarise(
    p_Stage           = extract_p(dplyr::pick(dplyr::everything()), "^Stage$"),
    p_Treatment       = extract_p(dplyr::pick(dplyr::everything()), "^Treatment$"),
    p_Stage_Treatment = extract_p(dplyr::pick(dplyr::everything()), "Stage.*Treatment|Treatment.*Stage"),
    .groups = "drop"
  )

main_summary <- lmm_status_all %>%
  dplyr::left_join(anova_pvals, by = "feature")

# Add wide raw means
if (!is.null(raw_wide)) {
  main_summary <- dplyr::left_join(main_summary, raw_wide, by = "feature")
}

# Reorder columns: feature, model_type, n_obs, n_subjects, raw means, p-values, flags
base_cols <- c("feature", "model_type", "n_obs", "n_subjects",
               "p_Stage", "p_Treatment", "p_Stage_Treatment",
               "singular", "convergence_warn", "error_msg")
raw_cols  <- setdiff(names(main_summary), base_cols)
main_summary <- main_summary %>%
  dplyr::select(dplyr::any_of(c(base_cols[1:4],
                                 grep("raw_mean", raw_cols, value = TRUE),
                                 base_cols[5:length(base_cols)])))

save_xlsx(main_summary,
          file.path(dir_tables, "main_lmm_summary_for_manuscript.xlsx"),
          "lmm_summary")
log_msg("Saved: main_lmm_summary_for_manuscript.xlsx")

# ── 2. LS means table ─────────────────────────────────────────────────────────
# Add n_obs and n_subjects from raw summary
raw_n <- lmm_raw_all %>%
  dplyr::select(feature, Treatment, Stage, n_squeals, n_subjects)

lsm_clean <- lmm_lsmeans_all %>%
  dplyr::rename_with(~ dplyr::case_when(
    . == "emmean"    ~ "ls_mean",
    . == "SE"        ~ "se",
    . == "df"        ~ "df",
    . == "lower.CL"  ~ "lower_95_CI",
    . == "upper.CL"  ~ "upper_95_CI",
    TRUE             ~ .
  )) %>%
  dplyr::mutate(
    CI_95_formatted = sprintf("[%.4f, %.4f]", lower_95_CI, upper_95_CI)
  ) %>%
  dplyr::left_join(raw_n, by = c("feature", "Treatment", "Stage")) %>%
  dplyr::select(feature, Treatment, Stage, ls_mean, se, df,
                lower_95_CI, upper_95_CI, CI_95_formatted,
                n_squeals, n_subjects)

save_xlsx(lsm_clean,
          file.path(dir_tables, "lsmeans_for_manuscript.xlsx"),
          "lsmeans")
log_msg("Saved: lsmeans_for_manuscript.xlsx")

# ── 3. Raw descriptives table ─────────────────────────────────────────────────
raw_desc_clean <- lmm_raw_all %>%
  dplyr::mutate(
    mean_sd      = sprintf("%.4f ± %.4f", mean, sd),
    median_IQR   = sprintf("%.4f [%.4f]", median, iqr),
    min_max      = sprintf("%.4f – %.4f", min, max)
  ) %>%
  dplyr::select(feature, Treatment, Stage,
                n_squeals, n_subjects,
                mean_sd, median_IQR, min_max)

save_xlsx(raw_desc_clean,
          file.path(dir_tables, "raw_descriptives_for_manuscript.xlsx"),
          "raw_descriptives")
log_msg("Saved: raw_descriptives_for_manuscript.xlsx")

# ── 4. Statistical methods draft ──────────────────────────────────────────────
n_feat    <- length(feature_cols)
n_full    <- sum(lmm_status_all$model_type == "full_interaction",  na.rm = TRUE)
n_fallbk  <- sum(lmm_status_all$model_type == "additive_fallback", na.rm = TRUE)
n_failed  <- sum(lmm_status_all$model_type == "failed",            na.rm = TRUE)

methods_text <- paste0(
  "# Statistical Methods\n\n",
  "Acoustic parameter values were analyzed at the squeal level (", nrow(dat_clean),
  " squeal events across ", dplyr::n_distinct(dat_clean$Subject), " pig pairs).\n\n",
  "Time was categorized into four stages: Pre, Early Post, Mid Post, and Late Post, ",
  "reflecting the recording schedule. Stages were treated as an unordered categorical factor ",
  "because recordings were sparse and irregularly spaced across pigs and weeks.\n\n",
  "For each of the ", n_feat, " acoustic features, a linear mixed-effects model was fit with ",
  "Stage, Treatment (S vs U), and their interaction (Stage × Treatment) as fixed effects, ",
  "and pig pair (Subject) as a random intercept to account for within-pair correlation ",
  "arising from repeated measurements. Models were fit using the lme4/lmerTest packages ",
  "in R (REML estimation; Satterthwaite degrees-of-freedom approximation). ",
  "Of ", n_feat, " features: ", n_full, " were fit with the full Stage × Treatment interaction model; ",
  n_fallbk, " required an additive Stage + Treatment fallback (interaction term could not be estimated); ",
  "and ", n_failed, " models failed entirely.\n\n",
  "Raw means were computed for direct interpretability. ",
  "Least-squares (LS) means and 95% confidence intervals were estimated from each model ",
  "using the emmeans package to support fair comparisons in the unbalanced dataset.\n\n",
  "Type III fixed-effect tests (F-statistics with Satterthwaite df) were used for primary inference, ",
  "with emphasis on the Stage × Treatment interaction p-value as the key test of whether ",
  "treatment groups differ across recovery stages. Exploratory pairwise post-hoc contrasts ",
  "(emmeans, Benjamini-Hochberg adjustment) were computed as secondary outputs.\n\n",
  "Kruskal-Wallis tests (Treatment effect within each Stage) and Wilcoxon rank-sum tests ",
  "(Pre vs each post stage within each pig pair) were performed as secondary/descriptive analyses only. ",
  "These nonparametric tests assume independent observations and do not fully account for ",
  "repeated measurements within pig pair; they should not be interpreted as primary evidence ",
  "of treatment effects.\n\n",
  "All analyses were performed in R version ", R.version$major, ".", R.version$minor,
  ". Significance level α = 0.05 was used for primary inference.\n"
)

writeLines(methods_text,
           file.path(dir_tables, "statistical_methods_draft.md"))
log_msg("Saved: statistical_methods_draft.md")

message("04_reporting_tables.R complete")
