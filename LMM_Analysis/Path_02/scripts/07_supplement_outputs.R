# 07_supplement_outputs.R
# Fills gaps identified from second prompt:
#   1. BH + BY adjusted p-values across all 50 features (Stage, Treatment, Interaction)
#   2. AIC, BIC, logLik, random intercept variance, residual variance in main summary
#   3. SE + 95% CI added to raw means table
#   4. Dedicated lmm_interaction_pvalue_adjustments table
#   5. analysis_dataset.csv
#   6. feature_inventory.csv
#   7. pair_level_wilcoxon_summary.csv (from existing Wilcoxon output)
#   8. model_failures.csv
#   9. manuscript_ready/ folder with final consolidated tables
#  10. Figures: treatment×stage count bar chart, faceted top-features plot, PDF copies

`%+%` <- function(a, b) paste0(a, b)

if (!exists("project_root")) {
  source("scripts/00_setup.R")
}
if (!exists("dat_clean")) {
  source("scripts/01_data_audit.R")
}

dir_manuscript <- file.path(project_root, "results", "manuscript_ready")
dir_pair       <- file.path(project_root, "results", "descriptive_pair_level")
dir_rlogs      <- file.path(project_root, "results", "logs")

for (d in c(dir_manuscript, dir_pair, dir_rlogs)) {
  if (!dir.exists(d)) dir.create(d, recursive = TRUE)
}

# ── Load existing tables ───────────────────────────────────────────────────────
main_sum <- readxl::read_excel(file.path(dir_tables, "main_lmm_summary_for_manuscript.xlsx"))
# Strip any enrichment columns from a previous run of this script to ensure idempotency
base_keep <- c("feature", "model_type", "n_obs", "n_subjects", "singular",
               "convergence_warn", "error_msg",
               "p_Stage", "p_Treatment", "p_Stage_Treatment",
               grep("^raw_mean_", names(main_sum), value = TRUE))
main_sum <- dplyr::select(main_sum, dplyr::any_of(base_keep))
raw_all  <- readxl::read_excel(file.path(dir_tables, "raw_means_by_stage_treatment_all_features.xlsx"))
lsm_all  <- readxl::read_excel(file.path(dir_tables, "lmm_lsmeans_stage_treatment_all_features.xlsx"))
status   <- readxl::read_excel(file.path(dir_tables, "lmm_model_status_log.xlsx"))
ranef_df <- readxl::read_excel(file.path(dir_tables, "lmm_random_effects_all_features.xlsx"))
wx_all   <- readxl::read_excel(file.path(dir_tables, "secondary_wilcoxon_pre_vs_post_within_subject.xlsx"))

# ── 1. BH + BY adjusted p-values across 50 features ──────────────────────────
adj_pvals <- main_sum %>%
  dplyr::select(feature, p_Stage, p_Treatment, p_Stage_Treatment) %>%
  dplyr::mutate(
    p_stage_adj_BH       = p.adjust(p_Stage,           method = "BH"),
    p_stage_adj_BY       = p.adjust(p_Stage,           method = "BY"),
    p_treatment_adj_BH   = p.adjust(p_Treatment,       method = "BH"),
    p_treatment_adj_BY   = p.adjust(p_Treatment,       method = "BY"),
    p_interaction_adj_BH = p.adjust(p_Stage_Treatment, method = "BH"),
    p_interaction_adj_BY = p.adjust(p_Stage_Treatment, method = "BY")
  )

# Save dedicated interaction p-value adjustment table
writexl::write_xlsx(adj_pvals,
                    file.path(dir_tables, "lmm_interaction_pvalue_adjustments.xlsx"))
write.csv(adj_pvals,
          file.path(dir_tables, "lmm_interaction_pvalue_adjustments.csv"),
          row.names = FALSE)
log_msg("Saved: lmm_interaction_pvalue_adjustments")

# ── 2. AIC, BIC, logLik, variance components from model objects ───────────────
model_files <- list.files(dir_models, pattern = "\\.rds$", full.names = TRUE)

fit_stats <- purrr::map_dfr(model_files, function(mf) {
  feat <- gsub("_lmm\\.rds$", "", basename(mf))
  match_idx <- which(safe_name(feature_cols) == feat)
  orig <- if (length(match_idx) > 0) feature_cols[match_idx[1]] else feat

  m <- tryCatch(readRDS(mf), error = function(e) NULL)
  if (is.null(m) || !inherits(m, "lmerMod")) return(NULL)

  vc  <- tryCatch(as.data.frame(lme4::VarCorr(m)), error = function(e) NULL)
  ri_var  <- if (!is.null(vc)) vc$vcov[vc$grp == "Subject"][1] else NA_real_
  res_var <- if (!is.null(vc)) vc$vcov[vc$grp == "Residual"][1] else NA_real_

  tibble::tibble(
    feature            = orig,
    AIC                = AIC(m),
    BIC                = BIC(m),
    logLik             = as.numeric(logLik(m)),
    random_int_var     = ri_var,
    residual_var       = res_var
  )
})

# ── 3. Merge into enriched main summary ───────────────────────────────────────
main_enriched <- main_sum %>%
  dplyr::left_join(adj_pvals %>%
                     dplyr::select(feature, dplyr::starts_with("p_stage_adj"),
                                   dplyr::starts_with("p_treatment_adj"),
                                   dplyr::starts_with("p_interaction_adj")),
                   by = "feature") %>%
  dplyr::left_join(fit_stats, by = "feature")

writexl::write_xlsx(main_enriched,
                    file.path(dir_tables, "main_lmm_summary_for_manuscript.xlsx"))
log_msg("Updated: main_lmm_summary_for_manuscript.xlsx (added adj p-values + AIC/BIC/logLik)")

# ── 4. Raw means + SE + 95% CI ────────────────────────────────────────────────
raw_enriched <- raw_all %>%
  dplyr::mutate(
    se       = sd / sqrt(n_squeals),
    ci95_low = mean - qt(0.975, df = pmax(n_squeals - 1, 1)) * se,
    ci95_high= mean + qt(0.975, df = pmax(n_squeals - 1, 1)) * se
  )

writexl::write_xlsx(raw_enriched,
                    file.path(dir_tables, "raw_means_by_stage_treatment_all_features.xlsx"))
write.csv(raw_enriched,
          file.path(dir_tables, "raw_feature_summary_by_stage_treatment.csv"),
          row.names = FALSE)
log_msg("Updated: raw_means_by_stage_treatment_all_features.xlsx (added SE, 95% CI)")

# ── 5. analysis_dataset.csv ───────────────────────────────────────────────────
write.csv(dat_clean,
          file.path(dir_tables, "analysis_dataset.csv"),
          row.names = FALSE)
log_msg("Saved: analysis_dataset.csv")

# ── 6. feature_inventory.csv ──────────────────────────────────────────────────
feat_inv <- tibble::tibble(
  feature_index = seq_along(feature_cols),
  feature       = feature_cols,
  n_nonmissing  = purrr::map_int(feature_cols, ~ sum(!is.na(dat_clean[[.x]]))),
  n_missing     = purrr::map_int(feature_cols, ~ sum( is.na(dat_clean[[.x]]))),
  pct_missing   = round(100 * n_missing / nrow(dat_clean), 2)
)
write.csv(feat_inv,
          file.path(dir_tables, "feature_inventory.csv"),
          row.names = FALSE)
log_msg("Saved: feature_inventory.csv")

# ── 7. pair_level_wilcoxon_summary.csv ────────────────────────────────────────
write.csv(wx_all,
          file.path(dir_pair, "pair_level_wilcoxon_summary.csv"),
          row.names = FALSE)
log_msg("Saved: pair_level_wilcoxon_summary.csv")

# ── 8. model_failures.csv ─────────────────────────────────────────────────────
failures <- status %>% dplyr::filter(model_type == "failed" | !is.na(error_msg))
write.csv(failures,
          file.path(dir_rlogs, "model_failures.csv"),
          row.names = FALSE)
log_msg("Saved: model_failures.csv (" %+% nrow(failures) %+% " entries)")

# ── 9. manuscript_ready/ tables ───────────────────────────────────────────────
# Table_LMM_Main_Results.xlsx: main model summary + adjusted p-values
manu_main <- main_enriched %>%
  dplyr::select(feature, model_type, n_obs, n_subjects,
                AIC, BIC, logLik,
                random_int_var, residual_var,
                p_Stage, p_Treatment, p_Stage_Treatment,
                p_stage_adj_BH, p_stage_adj_BY,
                p_treatment_adj_BH, p_treatment_adj_BY,
                p_interaction_adj_BH, p_interaction_adj_BY,
                singular, convergence_warn)

writexl::write_xlsx(manu_main,
                    file.path(dir_manuscript, "Table_LMM_Main_Results.xlsx"))
log_msg("Saved: Table_LMM_Main_Results.xlsx")

# Table_Raw_and_LSMeans.xlsx: two sheets
lsm_clean <- lsm_all %>%
  dplyr::rename_with(~ dplyr::case_when(
    . == "emmean"   ~ "ls_mean",
    . == "lower.CL" ~ "lower_95_CI",
    . == "upper.CL" ~ "upper_95_CI",
    TRUE            ~ .
  ))

openxlsx::write.xlsx(
  list(
    "LS_means"  = lsm_clean,
    "Raw_means" = raw_enriched
  ),
  file = file.path(dir_manuscript, "Table_Raw_and_LSMeans.xlsx"),
  overwrite = TRUE
)
log_msg("Saved: Table_Raw_and_LSMeans.xlsx")

# Methods_Statistical_Analysis.txt
methods_txt <- paste0(
  "Statistical Analysis\n",
  strrep("-", 60), "\n\n",
  "Acoustic parameter values were analyzed at the squeal level. ",
  "For each of the ", length(feature_cols), " acoustic features, linear mixed-effects regression ",
  "was performed using the lme4/lmerTest packages in R (version ", R.version$major, ".", R.version$minor, "). ",
  "Fixed effects were Stage (categorical: Pre, Early Post, Mid Post, Late Post), ",
  "Treatment (S vs U), and their interaction (Stage × Treatment). ",
  "Pig_ID (pig pair) was included as a random intercept to account for repeated measurements ",
  "within the same pig pair, as individual squeals from the same pair are not independent.\n\n",
  "Stage was modeled as an unordered categorical factor to align with the clinically defined ",
  "perioperative periods, because recordings were sparse and irregularly spaced across weeks. ",
  "Models were estimated using restricted maximum likelihood (REML) with Satterthwaite ",
  "degrees-of-freedom approximation for fixed-effect tests (Type III F-statistics).\n\n",
  "Raw observed means and standard errors are reported for direct interpretability. ",
  "Least-squares means (LS means / estimated marginal means) and 95% confidence intervals, ",
  "estimated using the emmeans package, are additionally reported because the dataset is ",
  "unbalanced with respect to pig pair and recording stage.\n\n",
  "The Stage × Treatment interaction p-value is the primary inferential focus, ",
  "addressing whether the trajectory of acoustic feature change across stages differs ",
  "between treatment groups. Omnibus p-values for all three fixed effects were adjusted ",
  "for multiple comparisons across ", length(feature_cols), " features using both ",
  "Benjamini-Hochberg (BH) and Benjamini-Yekutieli (BY) procedures.\n\n",
  "Exploratory post-hoc pairwise contrasts (Treatment S vs U within each Stage; ",
  "Stage comparisons within each Treatment) were computed from the model-based marginal means ",
  "using BH adjustment and are reported as secondary outputs.\n\n",
  "Pair-level Wilcoxon rank-sum analyses (Pre vs each post-operative stage within each pig pair) ",
  "are included as descriptive/supportive analyses only. These tests assume independent ",
  "observations and therefore do not fully account for the repeated-measures structure; ",
  "they should not be interpreted as primary evidence of treatment effects.\n\n",
  "Models that failed to converge or exhibited singular fits (random-effect variance at the ",
  "boundary) are flagged in the results. Singular fits were treated as a diagnostic indicator ",
  "and results interpreted cautiously, as they may reflect insufficient variation at the ",
  "pig-pair level for some features.\n"
)
writeLines(methods_txt,
           file.path(dir_manuscript, "Methods_Statistical_Analysis.txt"))
log_msg("Saved: Methods_Statistical_Analysis.txt")

# ── 10a. Count bar chart: Treatment × Stage ────────────────────────────────────
cnt_trt_stage <- dat_clean %>%
  dplyr::count(Treatment, Stage) %>%
  dplyr::mutate(Stage = factor(Stage, levels = stage_levels))

p_counts <- ggplot2::ggplot(cnt_trt_stage,
                             ggplot2::aes(x = Stage, y = n, fill = Treatment)) +
  ggplot2::geom_col(position = "dodge", colour = "white", width = 0.7) +
  ggplot2::geom_text(ggplot2::aes(label = n),
                     position = ggplot2::position_dodge(width = 0.7),
                     vjust = -0.4, size = 3.2) +
  ggplot2::scale_fill_manual(values = c(S = "#1b7837", U = "#c0392b")) +
  ggplot2::labs(title = "Squeal counts by Treatment × Stage",
                x = "Stage", y = "N squeals", fill = "Treatment") +
  ggplot2::theme_bw(base_size = 11) +
  ggplot2::theme(panel.grid.minor = ggplot2::element_blank())

ggplot2::ggsave(file.path(dir_figures,        "treatment_stage_counts_bar.png"),
                p_counts, width = 7, height = 4.5, dpi = 150)
ggplot2::ggsave(file.path(dir_figures,        "treatment_stage_counts_bar.pdf"),
                p_counts, width = 7, height = 4.5)
ggplot2::ggsave(file.path(dir_figures_lsmean, "treatment_stage_counts_bar.png"),
                p_counts, width = 7, height = 4.5, dpi = 150)
ggplot2::ggsave(file.path(dir_figures_lsmean, "treatment_stage_counts_bar.pdf"),
                p_counts, width = 7, height = 4.5)
log_msg("Saved: treatment_stage_counts_bar.png/pdf (Original + LS_Mean)")

# ── 10b. PDF copies of key trajectory plots ────────────────────────────────────
# Save PDF versions for the 5 specific features from the second prompt
key_feats <- intersect(c("ucla_f0", "ucla_jitter", "ucla_shimmer", "ucla_q50", "ucla_flux"),
                       feature_cols)

pal <- c(S = "#1b7837", U = "#c0392b")
dodge <- ggplot2::position_dodge(width = 0.25)

for (feat in key_feats) {
  raw_sub <- raw_enriched %>%
    dplyr::filter(feature == feat) %>%
    dplyr::mutate(Stage = factor(Stage, levels = stage_levels),
                  ymin_raw = mean - se, ymax_raw = mean + se)

  lsm_sub <- lsm_clean %>%
    dplyr::filter(feature == feat) %>%
    dplyr::mutate(Stage = factor(Stage, levels = stage_levels))

  p <- ggplot2::ggplot() +
    ggplot2::geom_line(data = raw_sub,
                       ggplot2::aes(x = Stage, y = mean, colour = Treatment, group = Treatment),
                       linewidth = 0.9, linetype = "solid", position = dodge) +
    ggplot2::geom_errorbar(data = raw_sub,
                           ggplot2::aes(x = Stage, ymin = ymin_raw, ymax = ymax_raw,
                                        colour = Treatment, group = Treatment),
                           width = 0.15, linewidth = 0.7, position = dodge) +
    ggplot2::geom_point(data = raw_sub,
                        ggplot2::aes(x = Stage, y = mean, colour = Treatment,
                                     group = Treatment, shape = "Raw mean ± SE"),
                        size = 3, position = dodge) +
    ggplot2::geom_line(data = lsm_sub,
                       ggplot2::aes(x = Stage, y = ls_mean, colour = Treatment, group = Treatment),
                       linewidth = 0.8, linetype = "dashed", position = dodge) +
    ggplot2::geom_errorbar(data = lsm_sub,
                           ggplot2::aes(x = Stage, ymin = lower_95_CI, ymax = upper_95_CI,
                                        colour = Treatment, group = Treatment),
                           width = 0.10, linewidth = 0.6, position = dodge) +
    ggplot2::geom_point(data = lsm_sub,
                        ggplot2::aes(x = Stage, y = ls_mean, colour = Treatment,
                                     group = Treatment, shape = "LS mean ± 95% CI"),
                        size = 3, position = dodge) +
    ggplot2::scale_colour_manual(values = pal, name = "Treatment") +
    ggplot2::scale_shape_manual(values = c("Raw mean ± SE" = 16, "LS mean ± 95% CI" = 17), name = "") +
    ggplot2::labs(title = feat,
                  subtitle = "Solid/circles = raw mean ± SE   |   Dashed/triangles = LS mean ± 95% CI",
                  x = "Stage", y = "Value") +
    ggplot2::theme_bw(base_size = 11) +
    ggplot2::theme(legend.position = "bottom", panel.grid.minor = ggplot2::element_blank())

  ggplot2::ggsave(file.path(dir_figures, safe_name(feat) %+% "_trajectory.pdf"),
                  p, width = 7, height = 4.5)
}
log_msg("Saved: PDF trajectory plots for key features (" %+% length(key_feats) %+% ")")

# ── 10c. Faceted plot: top significant interaction features ────────────────────
top_feats <- main_enriched %>%
  dplyr::filter(!is.na(p_Stage_Treatment)) %>%
  dplyr::arrange(p_Stage_Treatment) %>%
  dplyr::slice_head(n = 9) %>%
  dplyr::pull(feature)

if (length(top_feats) > 0) {
  raw_top <- raw_enriched %>%
    dplyr::filter(feature %in% top_feats) %>%
    dplyr::mutate(Stage   = factor(Stage, levels = stage_levels),
                  feature = factor(feature, levels = top_feats),
                  ymin    = mean - se,
                  ymax    = mean + se)

  p_facet <- ggplot2::ggplot(raw_top,
                              ggplot2::aes(x = Stage, y = mean,
                                           colour = Treatment, group = Treatment)) +
    ggplot2::geom_line(linewidth = 0.8, position = dodge) +
    ggplot2::geom_errorbar(ggplot2::aes(ymin = ymin, ymax = ymax),
                           width = 0.2, linewidth = 0.6, position = dodge) +
    ggplot2::geom_point(size = 2.5, position = dodge) +
    ggplot2::facet_wrap(~ feature, scales = "free_y", ncol = 3) +
    ggplot2::scale_colour_manual(values = pal, name = "Treatment") +
    ggplot2::labs(title = "Top 9 features by Stage × Treatment interaction p-value",
                  subtitle = "Raw mean ± SE",
                  x = "Stage", y = "Value") +
    ggplot2::theme_bw(base_size = 9) +
    ggplot2::theme(axis.text.x  = ggplot2::element_text(angle = 30, hjust = 1),
                   strip.text   = ggplot2::element_text(size = 7),
                   legend.position = "bottom",
                   panel.grid.minor = ggplot2::element_blank())

  ggplot2::ggsave(file.path(dir_figures,        "top_features_interaction_faceted.png"),
                  p_facet, width = 10, height = 9, dpi = 150)
  ggplot2::ggsave(file.path(dir_figures,        "top_features_interaction_faceted.pdf"),
                  p_facet, width = 10, height = 9)
  ggplot2::ggsave(file.path(dir_figures_lsmean, "top_features_interaction_faceted.png"),
                  p_facet, width = 10, height = 9, dpi = 150)
  ggplot2::ggsave(file.path(dir_figures_lsmean, "top_features_interaction_faceted.pdf"),
                  p_facet, width = 10, height = 9)
  log_msg("Saved: top_features_interaction_faceted.png/pdf (Original + LS_Mean)")
}

# ── 11. Missing CSV exports (Prompt 2 specifies these by name) ────────────────

# squeal_counts_by_pig_stage.csv and squeal_counts_by_treatment_stage.csv
counts_pig_stage <- dat_clean %>%
  dplyr::count(Subject, Treatment, Stage, name = "n_squeals") %>%
  dplyr::mutate(Stage = factor(Stage, levels = stage_levels)) %>%
  dplyr::arrange(Treatment, Subject, Stage)

counts_trt_stage <- dat_clean %>%
  dplyr::count(Treatment, Stage, name = "n_squeals") %>%
  dplyr::mutate(Stage = factor(Stage, levels = stage_levels)) %>%
  dplyr::arrange(Treatment, Stage)

write.csv(counts_pig_stage,
          file.path(dir_tables, "squeal_counts_by_pig_stage.csv"),     row.names = FALSE)
write.csv(counts_trt_stage,
          file.path(dir_tables, "squeal_counts_by_treatment_stage.csv"), row.names = FALSE)
log_msg("Saved: squeal_counts_by_pig_stage.csv, squeal_counts_by_treatment_stage.csv")

# lmm_fixed_effects_summary.csv
fixed_df <- readxl::read_excel(file.path(dir_tables, "lmm_fixed_effects_all_features.xlsx"))
write.csv(fixed_df, file.path(dir_tables, "lmm_fixed_effects_summary.csv"), row.names = FALSE)
log_msg("Saved: lmm_fixed_effects_summary.csv")

# lmm_model_diagnostics.csv
diag_df <- readxl::read_excel(file.path(dir_tables, "lmm_diagnostics_summary.xlsx"))
write.csv(diag_df, file.path(dir_tables, "lmm_model_diagnostics.csv"), row.names = FALSE)
log_msg("Saved: lmm_model_diagnostics.csv")

# emmeans_stage_treatment.csv
emm_df <- readxl::read_excel(file.path(dir_tables, "lmm_lsmeans_stage_treatment_all_features.xlsx"))
write.csv(emm_df, file.path(dir_tables, "emmeans_stage_treatment.csv"), row.names = FALSE)
log_msg("Saved: emmeans_stage_treatment.csv")

# contrasts_treatment_within_stage.csv and contrasts_stage_within_treatment.csv
pw_trt  <- readxl::read_excel(file.path(dir_tables, "lmm_pairwise_treatment_within_stage.xlsx"))
pw_stg  <- readxl::read_excel(file.path(dir_tables, "lmm_pairwise_stage_within_treatment.xlsx"))
write.csv(pw_trt, file.path(dir_tables, "contrasts_treatment_within_stage.csv"),  row.names = FALSE)
write.csv(pw_stg, file.path(dir_tables, "contrasts_stage_within_treatment.csv"),  row.names = FALSE)
log_msg("Saved: contrasts_treatment_within_stage.csv, contrasts_stage_within_treatment.csv")

message("07_supplement_outputs.R complete")
message("  Adjusted p-values added to main summary and saved separately")
message("  AIC/BIC/logLik/variance components added to main summary")
message("  SE and 95% CI added to raw means table")
message("  Manuscript-ready folder populated: ", dir_manuscript)
message("  Count bar chart, faceted plot, PDF key-feature plots saved")
message("  All missing CSV files exported")
