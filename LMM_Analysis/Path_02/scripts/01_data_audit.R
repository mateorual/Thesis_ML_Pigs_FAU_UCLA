# 01_data_audit.R — Read data, validate structure, summarise missingness and counts

if (!exists("project_root")) {
  source("scripts/00_setup.R")
}

# ── 1. Read data ───────────────────────────────────────────────────────────────
raw <- readxl::read_excel(input_file, sheet = "Features_Data")
log_msg("Read input: rows = " %+% nrow(raw) %+% ", cols = " %+% ncol(raw))

# ── 2. Rename Pig_ID → Subject ────────────────────────────────────────────────
# Input column is Pig_ID; standardise to Subject for all downstream scripts
if ("Pig_ID" %in% names(raw) && !"Subject" %in% names(raw)) {
  raw <- dplyr::rename(raw, Subject = Pig_ID)
  log_msg("Renamed Pig_ID to Subject")
}

# ── 3. Validate metadata columns ──────────────────────────────────────────────
expected_meta <- c("Squeal", "Subject", "Treatment", "Week", "Stage")
missing_meta  <- setdiff(expected_meta, names(raw))
if (length(missing_meta) > 0) {
  stop("Missing expected metadata columns: ", paste(missing_meta, collapse = ", "))
}
log_msg("Metadata columns validated: OK")

# ── 4. Identify acoustic feature columns ──────────────────────────────────────
meta_pos      <- match(expected_meta, names(raw))
feature_cols  <- names(raw)[seq(max(meta_pos) + 1, ncol(raw))]
n_features    <- length(feature_cols)
if (n_features != 50) {
  log_msg(paste0("WARNING: expected 50 feature columns, found ", n_features))
  message("WARNING: expected 50 feature columns, found ", n_features)
}
log_msg(paste0("Feature columns identified: ", n_features))

# ── 5. Type conversions ────────────────────────────────────────────────────────
dat <- raw %>%
  dplyr::mutate(
    Subject   = factor(Subject),
    Treatment = factor(Treatment, levels = treatment_levels),
    Stage     = factor(Stage,     levels = stage_levels),
    Week      = as.integer(Week)
  )

# Verify no unexpected Treatment/Stage values
bad_trt   <- setdiff(levels(dat$Treatment), treatment_levels)
bad_stage <- setdiff(levels(dat$Stage),     stage_levels)
if (length(bad_trt)   > 0) log_msg("Unexpected Treatment values: " %+% paste(bad_trt,   collapse=","))
if (length(bad_stage) > 0) log_msg("Unexpected Stage values: "     %+% paste(bad_stage, collapse=","))

# Save cleaned data to global env for downstream scripts
dat_clean    <<- dat
feature_cols <<- feature_cols

# ── 6. Missingness summary ─────────────────────────────────────────────────────
miss_summary <- purrr::map_dfr(feature_cols, function(f) {
  x <- dat[[f]]
  tibble::tibble(
    Feature       = f,
    n_total       = length(x),
    n_missing     = sum(is.na(x)),
    pct_missing   = round(100 * sum(is.na(x)) / length(x), 2),
    n_nonmissing  = sum(!is.na(x))
  )
})
save_xlsx(miss_summary,
          file.path(dir_tables, "feature_missingness.xlsx"),
          "missingness")
log_msg("Saved: feature_missingness.xlsx")

# ── 7. Sample-size summaries ───────────────────────────────────────────────────
cnt_by_treatment <- dat %>%
  dplyr::count(Treatment, name = "n_squeals")

cnt_by_subject <- dat %>%
  dplyr::count(Subject, Treatment, name = "n_squeals")

cnt_trt_stage <- dat %>%
  dplyr::count(Treatment, Stage, name = "n_squeals")

cnt_subj_stage <- dat %>%
  dplyr::count(Subject, Treatment, Stage, name = "n_squeals")

cnt_weeks_subj_stage <- dat %>%
  dplyr::group_by(Subject, Stage) %>%
  dplyr::summarise(n_distinct_weeks = dplyr::n_distinct(Week), .groups = "drop")

audit_list <- list(
  by_treatment       = cnt_by_treatment,
  by_subject         = cnt_by_subject,
  by_treatment_stage = cnt_trt_stage,
  by_subject_stage   = cnt_subj_stage,
  weeks_by_subj_stage = cnt_weeks_subj_stage
)
openxlsx::write.xlsx(audit_list,
                     file = file.path(dir_tables, "data_audit_counts.xlsx"),
                     overwrite = TRUE)
log_msg("Saved: data_audit_counts.xlsx")

# ── 8. Heatmap: squeals by Subject × Stage ────────────────────────────────────
p_heatmap <- ggplot2::ggplot(cnt_subj_stage,
                              ggplot2::aes(x = Stage, y = Subject, fill = n_squeals)) +
  ggplot2::geom_tile(colour = "white") +
  ggplot2::geom_text(ggplot2::aes(label = n_squeals), size = 3) +
  ggplot2::scale_fill_gradient(low = "#deebf7", high = "#08519c", name = "N squeals") +
  ggplot2::scale_x_discrete(limits = stage_levels) +
  ggplot2::labs(title = "Number of squeals by Subject × Stage",
                x = "Stage", y = "Subject") +
  ggplot2::theme_minimal(base_size = 11) +
  ggplot2::theme(axis.text.x = ggplot2::element_text(angle = 30, hjust = 1))

ggplot2::ggsave(file.path(dir_figures,        "subject_stage_counts_heatmap.png"),
                p_heatmap, width = 8, height = 5, dpi = 150)
ggplot2::ggsave(file.path(dir_figures_lsmean, "subject_stage_counts_heatmap.png"),
                p_heatmap, width = 8, height = 5, dpi = 150)
log_msg("Saved: subject_stage_counts_heatmap.png (Original + LS_Mean)")

message("01_data_audit.R complete — ", n_features, " features, ", nrow(dat), " rows")
