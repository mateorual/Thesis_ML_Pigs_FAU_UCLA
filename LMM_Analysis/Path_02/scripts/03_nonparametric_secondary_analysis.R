# 03_nonparametric_secondary_analysis.R
# SECONDARY / DESCRIPTIVE analyses only.
#
# IMPORTANT LIMITATION (encode in all output notes):
# These tests treat observations as independent and therefore do not fully
# account for repeated measurements within pig pair. They are included only
# as limited descriptive/secondary analyses; the primary inference comes
# from the LMM.

`%+%` <- function(a, b) paste0(a, b)

if (!exists("dat_clean")) {
  source("scripts/00_setup.R")
  source("scripts/01_data_audit.R")
}

independence_note <- paste0(
  "SECONDARY/DESCRIPTIVE ONLY. These tests treat observations as independent ",
  "and therefore do not fully account for repeated measurements within pig pair. ",
  "Primary inference comes from the LMM (script 02)."
)

# ── 1. Kruskal-Wallis: Treatment effect within each Stage ─────────────────────
# For each feature × stage, test H0: distribution is equal across S and U.
# Only run when both treatments are present with adequate data.

kw_rows <- list()

for (feat in feature_cols) {
  for (stg in stage_levels) {
    d_sub <- dat_clean %>%
      dplyr::filter(Stage == stg, !is.na(.data[[feat]])) %>%
      dplyr::select(Treatment, y = dplyr::all_of(feat))

    trts_present <- unique(d_sub$Treatment)
    if (length(trts_present) < 2) next

    n_per_trt <- table(d_sub$Treatment)
    if (any(n_per_trt < 3)) next   # skip if any group too small

    kw <- tryCatch(
      kruskal.test(y ~ Treatment, data = d_sub),
      error = function(e) NULL
    )
    if (is.null(kw)) next

    kw_rows[[length(kw_rows) + 1]] <- tibble::tibble(
      feature      = feat,
      stage        = stg,
      n_S          = as.integer(n_per_trt["S"]),
      n_U          = as.integer(n_per_trt["U"]),
      statistic    = kw$statistic,
      df           = kw$parameter,
      p_raw        = kw$p.value,
      note         = independence_note
    )
  }
}

kw_all <- dplyr::bind_rows(kw_rows)

# BH correction across all tests (exploratory)
if (nrow(kw_all) > 0) {
  kw_all$p_BH <- p.adjust(kw_all$p_raw, method = "BH")
  kw_all$label <- "secondary_exploratory"
  save_xlsx(kw_all,
            file.path(dir_tables, "secondary_kruskal_treatment_within_stage.xlsx"),
            "kruskal_wallis")
  log_msg("Saved: secondary_kruskal_treatment_within_stage.xlsx (" %+% nrow(kw_all) %+% " rows)")
}

# ── 2. Wilcoxon rank-sum: Pre vs each Post stage within each Subject ──────────
# For each feature × subject × comparison (Pre vs Early/Mid/Late Post),
# test whether value distributions differ.
# Both groups are independent samples of squeals from the same pig pair at
# different time stages — NOT paired at the squeal level; label accordingly.

post_stages <- c("Early Post", "Mid Post", "Late Post")
wx_rows     <- list()

for (feat in feature_cols) {
  for (subj in levels(dat_clean$Subject)) {
    for (post_stg in post_stages) {

      pre_vals  <- dat_clean %>%
        dplyr::filter(Subject == subj, Stage == "Pre",    !is.na(.data[[feat]])) %>%
        dplyr::pull(feat)
      post_vals <- dat_clean %>%
        dplyr::filter(Subject == subj, Stage == post_stg, !is.na(.data[[feat]])) %>%
        dplyr::pull(feat)

      if (length(pre_vals) < 3 || length(post_vals) < 3) next

      wx <- tryCatch(
        wilcox.test(pre_vals, post_vals, exact = FALSE),
        error = function(e) NULL
      )
      if (is.null(wx)) next

      wx_rows[[length(wx_rows) + 1]] <- tibble::tibble(
        feature       = feat,
        subject       = subj,
        comparison    = "Pre vs " %+% post_stg,
        n_pre         = length(pre_vals),
        n_post        = length(post_vals),
        W_statistic   = wx$statistic,
        p_raw         = wx$p.value,
        note          = independence_note
      )
    }
  }
}

wx_all <- dplyr::bind_rows(wx_rows)

# Benjamini-Yekutieli (BY) correction — valid under general dependency
if (nrow(wx_all) > 0) {
  wx_all$p_BY <- p.adjust(wx_all$p_raw, method = "BY")
  wx_all$label <- "secondary_exploratory"
  save_xlsx(wx_all,
            file.path(dir_tables, "secondary_wilcoxon_pre_vs_post_within_subject.xlsx"),
            "wilcoxon_pre_vs_post")
  log_msg("Saved: secondary_wilcoxon_pre_vs_post_within_subject.xlsx (" %+% nrow(wx_all) %+% " rows)")
}

message("03_nonparametric_secondary_analysis.R complete")
message("  Kruskal-Wallis tests : ", nrow(kw_all))
message("  Wilcoxon tests       : ", nrow(wx_all))
