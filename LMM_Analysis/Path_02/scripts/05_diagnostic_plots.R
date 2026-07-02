# 05_diagnostic_plots.R — Diagnostic plots and summary table for fitted LMMs

`%+%` <- function(a, b) paste0(a, b)

if (!exists("dat_clean")) {
  source("scripts/00_setup.R")
  source("scripts/01_data_audit.R")
}

if (!exists("lmm_status_all")) {
  lmm_status_all <- readxl::read_excel(file.path(dir_tables, "lmm_model_status_log.xlsx"))
}

diag_rows <- list()

# Features that have saved model files
model_files <- list.files(dir_models, pattern = "\\.rds$", full.names = TRUE)

for (mf in model_files) {
  feat <- gsub("_lmm\\.rds$", "", basename(mf))
  # Recover original feature name from safe_name (underscore may collide, but best effort)
  # Try exact match first, then fuzzy
  orig_feat <- feat
  match_idx <- which(safe_name(feature_cols) == feat)
  if (length(match_idx) > 0) orig_feat <- feature_cols[match_idx[1]]

  model <- tryCatch(readRDS(mf), error = function(e) NULL)
  if (is.null(model) || !inherits(model, "lmerMod")) next

  # Extract diagnostics
  resid_vals  <- residuals(model)
  fitted_vals <- fitted(model)
  is_sing     <- lme4::isSingular(model, tol = 1e-4)
  vc          <- tryCatch(as.data.frame(lme4::VarCorr(model)), error = function(e) NULL)
  ri_var      <- if (!is.null(vc)) vc$vcov[vc$grp == "Subject"][1] else NA_real_
  resid_sd    <- sd(resid_vals, na.rm = TRUE)
  n_obs       <- length(resid_vals)
  n_subj      <- tryCatch(
    dplyr::n_distinct(model@frame[["Subject"]]),
    error = function(e) NA_integer_
  )

  # Convergence warnings from status log
  status_row  <- lmm_status_all[lmm_status_all$feature == orig_feat, ]
  conv_warn   <- if (nrow(status_row) > 0) status_row$convergence_warn[1] else NA
  model_type  <- if (nrow(status_row) > 0) status_row$model_type[1]      else NA

  diag_rows[[orig_feat]] <- tibble::tibble(
    feature          = orig_feat,
    model_type       = model_type,
    singular_fit     = is_sing,
    convergence_warn = conv_warn,
    residual_sd      = resid_sd,
    random_int_var   = ri_var,
    n_obs            = n_obs,
    n_subjects       = n_subj,
    notes            = dplyr::case_when(
      is_sing  ~ "Singular fit: random-effect variance near boundary",
      conv_warn~ "Convergence warning: interpret estimates cautiously",
      TRUE     ~ ""
    )
  )

  # ── Diagnostic plots (4-panel) ──────────────────────────────────────────────
  png_path <- file.path(dir_diagnostics, safe_name(orig_feat) %+% "_diagnostics.png")

  tryCatch({
    png(png_path, width = 1200, height = 900, res = 120)
    par(mfrow = c(2, 2), mar = c(4, 4, 3, 1))

    # 1. Residuals vs Fitted
    plot(fitted_vals, resid_vals,
         xlab = "Fitted values", ylab = "Residuals",
         main = orig_feat %+% " — Residuals vs Fitted",
         pch = 20, cex = 0.6, col = "#2166ac80")
    abline(h = 0, col = "red", lty = 2)

    # 2. QQ plot
    qqnorm(resid_vals, main = "QQ Plot of Residuals",
           pch = 20, cex = 0.6, col = "#2166ac80")
    qqline(resid_vals, col = "red", lty = 2)

    # 3. Histogram of residuals
    hist(resid_vals, breaks = 30, col = "#deebf7", border = "white",
         main = "Histogram of Residuals",
         xlab = "Residuals", freq = FALSE)
    curve(dnorm(x, mean = mean(resid_vals), sd = sd(resid_vals)),
          add = TRUE, col = "red", lwd = 2)

    # 4. Random intercept BLUPs
    re <- tryCatch(lme4::ranef(model)$Subject[[1]], error = function(e) NULL)
    if (!is.null(re) && length(re) > 0) {
      subj_names <- rownames(lme4::ranef(model)$Subject)
      ord        <- order(re)
      dotchart(re[ord], labels = subj_names[ord],
               main = "Random Intercept BLUPs (Subject)",
               xlab = "BLUP estimate",
               pch = 19, color = "#08519c", cex = 0.8)
      abline(v = 0, col = "red", lty = 2)
    } else {
      plot.new(); title("Random intercept BLUPs unavailable")
    }

    dev.off()
  }, error = function(e) {
    if (dev.cur() > 1) dev.off()
    log_msg("Diagnostic plot failed for " %+% orig_feat %+% ": " %+% conditionMessage(e))
  })

  # ── DHARMa simulated residual diagnostics ────────────────────────────────────
  # DHARMa generates scaled simulation-based residuals for LMMs, which should
  # be uniformly distributed if the model is correctly specified — more
  # informative than raw residuals for mixed models.
  dharma_path <- file.path(dir_diagnostics, safe_name(orig_feat) %+% "_dharma.png")
  tryCatch({
    sim_res <- DHARMa::simulateResiduals(fittedModel = model, n = 500, plot = FALSE)
    png(dharma_path, width = 1000, height = 500, res = 120)
    par(mfrow = c(1, 2))
    DHARMa::plotQQunif(sim_res, main = orig_feat %+% " — DHARMa QQ")
    DHARMa::plotResiduals(sim_res, main = "Residuals vs Fitted (DHARMa)")
    dev.off()
  }, error = function(e) {
    if (dev.cur() > 1) dev.off()
    log_msg("DHARMa plot failed for " %+% orig_feat %+% ": " %+% conditionMessage(e))
  })

} # end model loop

# ── Save diagnostic summary table ─────────────────────────────────────────────
diag_summary <- dplyr::bind_rows(diag_rows)
if (nrow(diag_summary) > 0) {
  save_xlsx(diag_summary,
            file.path(dir_tables, "lmm_diagnostics_summary.xlsx"),
            "diagnostics")
  log_msg("Saved: lmm_diagnostics_summary.xlsx (" %+% nrow(diag_summary) %+% " rows)")
}

message("05_diagnostic_plots.R complete — plots saved for ", length(model_files), " models")
