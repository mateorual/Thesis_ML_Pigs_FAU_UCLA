# run_all.R — Run the full LMM pipeline in order
# Run with the working directory set to Path_02/ (open Path_02.Rproj, or
# setwd("<path to>/Path_02") first), e.g.: source("scripts/run_all.R")

t_start <- proc.time()

scripts_dir <- "scripts"

run_script <- function(name) {
  path <- file.path(scripts_dir, name)
  message("\n", strrep("=", 60))
  message("Running: ", name)
  message(strrep("=", 60))
  source(path, local = FALSE)
}

run_script("00_setup.R")
run_script("01_data_audit.R")
run_script("02_lmm_primary_analysis.R")
run_script("03_nonparametric_secondary_analysis.R")
run_script("04_reporting_tables.R")
run_script("05_diagnostic_plots.R")
run_script("06_trajectory_plots.R")
run_script("07_supplement_outputs.R")
run_script("08_spaghetti_plots.R")

# ── Final console summary ─────────────────────────────────────────────────────
elapsed <- round((proc.time() - t_start)["elapsed"])

n_feat     <- length(feature_cols)
n_full     <- sum(lmm_status_all$model_type == "full_interaction",  na.rm = TRUE)
n_fallback <- sum(lmm_status_all$model_type == "additive_fallback", na.rm = TRUE)
n_failed   <- sum(lmm_status_all$model_type == "failed",            na.rm = TRUE)

message("\n", strrep("=", 60))
message("PIPELINE COMPLETE  (", elapsed, " sec)")
message(strrep("=", 60))
message("Features detected            : ", n_feat)
message("Full interaction models      : ", n_full)
message("Additive fallback models     : ", n_fallback)
message("Failed models                : ", n_failed)
message("")
message("Manuscript tables → ", dir_tables)
message("  main_lmm_summary_for_manuscript.xlsx")
message("  lsmeans_for_manuscript.xlsx")
message("  raw_descriptives_for_manuscript.xlsx")
message("  statistical_methods_draft.md")
message("")
message("Diagnostic outputs → ", dir_diagnostics)
message("  lmm_diagnostics_summary.xlsx (in results/tables)")
message(strrep("=", 60))
