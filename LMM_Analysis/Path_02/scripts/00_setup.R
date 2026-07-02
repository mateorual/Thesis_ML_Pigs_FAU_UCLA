# 00_setup.R — Project configuration, package loading, and shared helpers

# ── Paths ─────────────────────────────────────────────────────────────────────
# project_root resolves to the current working directory, which must be this
# repository's Path_02/ folder. Opening Path_02.Rproj in RStudio sets this
# automatically; otherwise call setwd("<path to>/Path_02") first.
project_root <- normalizePath(".")
input_file   <- file.path(project_root, "Selected_Features_Groups_S_U_v1.xlsx")

dir_scripts     <- file.path(project_root, "scripts")
dir_results     <- file.path(project_root, "results")
dir_tables      <- file.path(project_root, "results", "tables")
dir_models      <- file.path(project_root, "results", "models")
dir_diagnostics       <- file.path(project_root, "results", "diagnostics")
dir_figures           <- file.path(project_root, "results", "figures", "Original")
dir_figures_lsmean    <- file.path(project_root, "results", "figures", "LS_Mean")
dir_figures_spaghetti <- file.path(project_root, "results", "figures", "Spaghetti")
dir_logs              <- file.path(project_root, "logs")

for (d in c(dir_scripts, dir_results, dir_tables, dir_models,
            dir_diagnostics, dir_figures, dir_figures_lsmean,
            dir_figures_spaghetti, dir_logs)) {
  if (!dir.exists(d)) dir.create(d, recursive = TRUE)
}

# ── Packages ──────────────────────────────────────────────────────────────────
required_pkgs <- c(
  "readxl", "writexl", "openxlsx",
  "dplyr", "tidyr", "purrr", "stringr", "forcats",
  "lme4", "lmerTest", "emmeans", "broom.mixed",
  "performance", "DHARMa",
  "ggplot2", "ggtext", "rstatix", "janitor"
)

for (pkg in required_pkgs) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    message("Installing missing package: ", pkg)
    install.packages(pkg, repos = "https://cloud.r-project.org")
  }
  suppressPackageStartupMessages(library(pkg, character.only = TRUE))
}

# ── Global options ─────────────────────────────────────────────────────────────
set.seed(42)
options(scipen = 10, digits = 6, warn = 1)

# Stage is modeled as a nominal (unordered) factor with Pre as the reference level.
# R default treatment contrasts (contr.treatment) are used so that:
#   - Pre is the baseline; each other stage is estimated as a deviation from Pre
#   - Temporal order is preserved only in plots, tables, and factor levels
#   - Do NOT use contr.sum here: sum-to-zero coding removes the Pre reference level
#     and makes coefficients less clinically interpretable
# The Stage:Treatment interaction F-test (primary inference) is identical
# under either contrast scheme; emmeans LS means are also contrast-coding invariant.

# ── Factor levels ──────────────────────────────────────────────────────────────
# Note: input column is Pig_ID; we rename to Subject throughout
metadata_cols  <- c("Squeal", "Subject", "Treatment", "Week", "Stage")
stage_levels   <- c("Pre", "Early Post", "Mid Post", "Late Post")
treatment_levels <- c("S", "U")

# ── Helper: safe filename ──────────────────────────────────────────────────────
safe_name <- function(x) {
  gsub("[^A-Za-z0-9_]", "_", x)
}

# ── Helper: fit LMM with tryCatch ──────────────────────────────────────────────
fit_lmm <- function(formula, data) {
  tryCatch(
    lmerTest::lmer(formula, data = data, REML = TRUE),
    error   = function(e) list(error = conditionMessage(e)),
    warning = function(w) {
      # Still attempt fit but capture warning
      withCallingHandlers(
        lmerTest::lmer(formula, data = data, REML = TRUE),
        warning = function(w2) {
          message("LMM warning: ", conditionMessage(w2))
          invokeRestart("muffleWarning")
        }
      )
    }
  )
}

# ── Helper: extract Type III ANOVA p-values ───────────────────────────────────
get_anova_pvals <- function(model) {
  tryCatch({
    at <- as.data.frame(anova(model, type = 3, ddf = "Satterthwaite"))
    at$term <- rownames(at)
    at
  }, error = function(e) NULL)
}

# ── Helper: save data frame to xlsx ───────────────────────────────────────────
save_xlsx <- function(df, path, sheet = "Sheet1") {
  tryCatch(
    writexl::write_xlsx(setNames(list(df), sheet), path),
    error = function(e) message("Could not save ", path, ": ", conditionMessage(e))
  )
}

# ── Helper: append to log file ────────────────────────────────────────────────
log_msg <- function(msg, log_file = file.path(dir_logs, "pipeline.log")) {
  ts <- format(Sys.time(), "%Y-%m-%d %H:%M:%S")
  cat(paste0("[", ts, "] ", msg, "\n"), file = log_file, append = TRUE)
}

# String concatenation shorthand used across all scripts
`%+%` <- function(a, b) paste0(a, b)

message("00_setup.R complete — project root: ", project_root)
