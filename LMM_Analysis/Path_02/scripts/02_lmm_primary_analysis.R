# 02_lmm_primary_analysis.R
# Primary LMM analysis: feature ~ Stage * Treatment + (1|Subject)
# Stage and Treatment are fixed effects (categorical); Subject is a random intercept.
# Stage*Treatment interaction is the key inferential term.
# LS means account for imbalance; raw means are reported for interpretability.

# ── Setup ─────────────────────────────────────────────────────────────────────
`%+%` <- function(a, b) paste0(a, b)

if (!exists("dat_clean")) {
  source("scripts/00_setup.R")
  source("scripts/01_data_audit.R")
}

# ── Containers ────────────────────────────────────────────────────────────────
list_anova        <- list()
list_fixed        <- list()
list_ranef        <- list()
list_lsmeans      <- list()
list_pw_trt       <- list()   # pairwise Treatment within Stage
list_pw_stage     <- list()   # pairwise Stage within Treatment
list_raw          <- list()
list_status       <- list()

# ── Main loop ─────────────────────────────────────────────────────────────────
for (feat in feature_cols) {

  status_row <- tibble::tibble(
    feature          = feat,
    model_type       = NA_character_,
    n_obs            = NA_integer_,
    n_subjects       = NA_integer_,
    singular         = NA,
    convergence_warn = NA,
    error_msg        = NA_character_
  )

  # -- Raw descriptives (always) -----------------------------------------------
  raw_desc <- dat_clean %>%
    dplyr::filter(!is.na(.data[[feat]])) %>%
    dplyr::group_by(Treatment, Stage) %>%
    dplyr::summarise(
      feature      = feat,
      n_squeals    = dplyr::n(),
      n_subjects   = dplyr::n_distinct(Subject),
      mean         = mean(.data[[feat]], na.rm = TRUE),
      sd           = sd(.data[[feat]],   na.rm = TRUE),
      median       = median(.data[[feat]], na.rm = TRUE),
      iqr          = IQR(.data[[feat]],  na.rm = TRUE),
      min          = min(.data[[feat]],  na.rm = TRUE),
      max          = max(.data[[feat]],  na.rm = TRUE),
      .groups      = "drop"
    ) %>%
    dplyr::select(feature, dplyr::everything())
  list_raw[[feat]] <- raw_desc

  # -- Prepare model data ------------------------------------------------------
  d <- dat_clean %>%
    dplyr::filter(!is.na(.data[[feat]]),
                  !is.na(Subject),
                  !is.na(Treatment),
                  !is.na(Stage)) %>%
    dplyr::mutate(y = .data[[feat]])

  n_obs  <- nrow(d)
  n_subj <- dplyr::n_distinct(d$Subject)
  n_stg  <- dplyr::n_distinct(d$Stage)
  n_trt  <- dplyr::n_distinct(d$Treatment)

  status_row$n_obs      <- n_obs
  status_row$n_subjects <- n_subj

  # Need at least 2 stages and both treatments for interaction model
  can_fit_full <- (n_stg >= 2) && (n_trt == 2) && (n_obs >= 10) && (n_subj >= 2)

  model       <- NULL
  model_type  <- "failed"
  conv_warn   <- FALSE
  warnings_captured <- character(0)

  # -- Fit full interaction model ----------------------------------------------
  if (can_fit_full) {
    model <- tryCatch({
      withCallingHandlers(
        lmerTest::lmer(y ~ Stage * Treatment + (1 | Subject), data = d, REML = TRUE),
        warning = function(w) {
          warnings_captured <<- c(warnings_captured, conditionMessage(w))
          invokeRestart("muffleWarning")
        }
      )
    }, error = function(e) {
      log_msg("FULL model error for " %+% feat %+% ": " %+% conditionMessage(e))
      NULL
    })

    if (!is.null(model) && inherits(model, "lmerMod")) {
      model_type <- "full_interaction"
    } else {
      model <- NULL
    }
  }

  # -- Fallback: additive model ------------------------------------------------
  if (is.null(model) && n_stg >= 2 && n_trt >= 1 && n_obs >= 6 && n_subj >= 2) {
    warnings_captured <- character(0)
    model <- tryCatch({
      withCallingHandlers(
        lmerTest::lmer(y ~ Stage + Treatment + (1 | Subject), data = d, REML = TRUE),
        warning = function(w) {
          warnings_captured <<- c(warnings_captured, conditionMessage(w))
          invokeRestart("muffleWarning")
        }
      )
    }, error = function(e) {
      log_msg("ADDITIVE model error for " %+% feat %+% ": " %+% conditionMessage(e))
      NULL
    })

    if (!is.null(model) && inherits(model, "lmerMod")) {
      model_type <- "additive_fallback"
      log_msg("Used additive fallback for: " %+% feat)
    } else {
      model <- NULL
    }
  }

  # -- If no model fits --------------------------------------------------------
  if (is.null(model)) {
    status_row$model_type    <- "failed"
    status_row$error_msg     <- "Both full and additive models failed"
    list_status[[feat]]      <- status_row
    log_msg("FAILED both models for: " %+% feat)
    next
  }

  conv_warn <- any(grepl("convergence|singular|failed to converge",
                          warnings_captured, ignore.case = TRUE))
  is_sing   <- lme4::isSingular(model, tol = 1e-4)

  status_row$model_type       <- model_type
  status_row$singular         <- is_sing
  status_row$convergence_warn <- conv_warn

  if (is_sing)   log_msg("Singular fit for: " %+% feat)
  if (conv_warn) log_msg("Convergence warning for: " %+% feat)

  # Save model object
  saveRDS(model, file.path(dir_models, safe_name(feat) %+% "_lmm.rds"))

  # -- Type III ANOVA ----------------------------------------------------------
  anova_tbl <- tryCatch({
    at <- as.data.frame(anova(model, type = 3, ddf = "Satterthwaite"))
    at$feature <- feat
    at$term    <- rownames(at)
    at
  }, error = function(e) {
    log_msg("ANOVA failed for " %+% feat %+% ": " %+% conditionMessage(e))
    NULL
  })
  if (!is.null(anova_tbl)) list_anova[[feat]] <- anova_tbl

  # -- Fixed effects -----------------------------------------------------------
  fixed_tbl <- tryCatch({
    ft <- as.data.frame(coef(summary(model)))
    ft$feature <- feat
    ft$term    <- rownames(ft)
    ft
  }, error = function(e) NULL)
  if (!is.null(fixed_tbl)) list_fixed[[feat]] <- fixed_tbl

  # -- Random effects ----------------------------------------------------------
  ranef_tbl <- tryCatch({
    vc <- as.data.frame(lme4::VarCorr(model))
    vc$feature <- feat
    vc
  }, error = function(e) NULL)
  if (!is.null(ranef_tbl)) list_ranef[[feat]] <- ranef_tbl

  # -- LS means: Stage × Treatment ---------------------------------------------
  lsm <- tryCatch({
    em  <- emmeans::emmeans(model, ~ Stage * Treatment)
    emdf <- as.data.frame(em)
    emdf$feature <- feat
    emdf
  }, error = function(e) {
    log_msg("emmeans failed for " %+% feat %+% ": " %+% conditionMessage(e))
    NULL
  })
  if (!is.null(lsm)) list_lsmeans[[feat]] <- lsm

  # -- Pairwise: Treatment within Stage (BH-adjusted, secondary/exploratory) ---
  pw_trt <- tryCatch({
    em_trt  <- emmeans::emmeans(model, ~ Treatment | Stage)
    ct      <- as.data.frame(emmeans::contrast(em_trt, method = "pairwise", adjust = "BH"))
    ct$feature <- feat
    ct$note    <- "BH-adjusted; exploratory post-hoc"
    ct
  }, error = function(e) NULL)
  if (!is.null(pw_trt)) list_pw_trt[[feat]] <- pw_trt

  # -- Pairwise: Stage within Treatment (BH-adjusted, secondary/exploratory) ---
  pw_stage <- tryCatch({
    em_stg  <- emmeans::emmeans(model, ~ Stage | Treatment)
    cs      <- as.data.frame(emmeans::contrast(em_stg, method = "pairwise", adjust = "BH"))
    cs$feature <- feat
    cs$note    <- "BH-adjusted; exploratory post-hoc"
    cs
  }, error = function(e) NULL)
  if (!is.null(pw_stage)) list_pw_stage[[feat]] <- pw_stage

  list_status[[feat]] <- status_row

} # end feature loop

# ── Bind and save all outputs ─────────────────────────────────────────────────

bind_save <- function(lst, path, sheet) {
  if (length(lst) == 0) return(invisible(NULL))
  df <- dplyr::bind_rows(lst)
  save_xlsx(df, path, sheet)
  log_msg("Saved: " %+% basename(path))
  df
}

anova_all  <- bind_save(list_anova,    file.path(dir_tables, "lmm_type3_anova_all_features.xlsx"),              "type3_anova")
fixed_all  <- bind_save(list_fixed,    file.path(dir_tables, "lmm_fixed_effects_all_features.xlsx"),            "fixed_effects")
ranef_all  <- bind_save(list_ranef,    file.path(dir_tables, "lmm_random_effects_all_features.xlsx"),           "random_effects")
lsm_all    <- bind_save(list_lsmeans,  file.path(dir_tables, "lmm_lsmeans_stage_treatment_all_features.xlsx"),  "lsmeans")
pw_trt_all <- bind_save(list_pw_trt,   file.path(dir_tables, "lmm_pairwise_treatment_within_stage.xlsx"),       "pw_treatment_within_stage")
pw_stg_all <- bind_save(list_pw_stage, file.path(dir_tables, "lmm_pairwise_stage_within_treatment.xlsx"),       "pw_stage_within_treatment")
raw_all    <- bind_save(list_raw,      file.path(dir_tables, "raw_means_by_stage_treatment_all_features.xlsx"),  "raw_means")
status_all <- bind_save(list_status,   file.path(dir_tables, "lmm_model_status_log.xlsx"),                      "model_status")

# Expose for downstream scripts
lmm_anova_all   <<- anova_all
lmm_lsmeans_all <<- lsm_all
lmm_raw_all     <<- raw_all
lmm_status_all  <<- status_all

n_full     <- sum(status_all$model_type == "full_interaction",  na.rm = TRUE)
n_fallback <- sum(status_all$model_type == "additive_fallback", na.rm = TRUE)
n_failed   <- sum(status_all$model_type == "failed",            na.rm = TRUE)

message("02_lmm_primary_analysis.R complete")
message("  Full interaction models : ", n_full)
message("  Additive fallback models: ", n_fallback)
message("  Failed models           : ", n_failed)
