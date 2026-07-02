# run_key_panel.R — standalone runner for the key-features LS-mean panel
# Generates key_features_lsmean_panel.pdf/.png in results/figures/LS_Mean/
# Run with the working directory set to Path_02/ (requires 02_lmm_primary_analysis.R
# to have already been run at least once, so that results/tables/ is populated).

project_root    <- normalizePath(".")
dir_tables      <- file.path(project_root, "results", "tables")
dir_figures_lsmean <- file.path(project_root, "results", "figures", "LS_Mean")

stage_levels <- c("Pre", "Early Post", "Mid Post", "Late Post")
pal          <- c(S = "#1b7837", U = "#c0392b")
dodge        <- ggplot2::position_dodge(width = 0.25)

for (pkg in c("readxl", "dplyr", "ggplot2", "patchwork")) {
  if (!requireNamespace(pkg, quietly = TRUE)) install.packages(pkg, repos = "https://cloud.r-project.org")
  suppressPackageStartupMessages(library(pkg, character.only = TRUE))
}

lsm_all <- readxl::read_excel(file.path(dir_tables, "lmm_lsmeans_stage_treatment_all_features.xlsx"))
if ("lower.CL" %in% names(lsm_all)) lsm_all <- dplyr::rename(lsm_all, lower_95_CI = lower.CL, upper_95_CI = upper.CL)
if ("emmean"   %in% names(lsm_all)) lsm_all <- dplyr::rename(lsm_all, ls_mean = emmean)

feature_cols <- unique(lsm_all$feature)

key_feats_units <- c(
  gfcc_4_std     = "a.u.",
  pncc_4_std     = "a.u.",
  s_entropy_mean = "a.u.",
  s_flux_mean    = "a.u.",
  Schlegel21_HNR = "dB"
)

key_feats_present <- intersect(names(key_feats_units), feature_cols)
message("Key features found: ", paste(key_feats_present, collapse = ", "))

make_key_panel <- function(feat) {
  unit_str <- key_feats_units[feat]
  y_label  <- paste0(feat, " (", unit_str, ")")

  lsm_sub <- lsm_all %>%
    dplyr::filter(feature == feat) %>%
    dplyr::mutate(Stage = factor(Stage, levels = stage_levels))

  ggplot2::ggplot(
    lsm_sub,
    ggplot2::aes(x = Stage, y = ls_mean, colour = Treatment, group = Treatment)
  ) +
    ggplot2::geom_line(linewidth = 0.9, position = dodge) +
    ggplot2::geom_errorbar(
      ggplot2::aes(ymin = lower_95_CI, ymax = upper_95_CI),
      width = 0.12, linewidth = 0.7, position = dodge
    ) +
    ggplot2::geom_point(size = 3, position = dodge) +
    ggplot2::scale_colour_manual(values = pal, name = "Treatment") +
    ggplot2::labs(x = "Stage", y = y_label) +
    ggplot2::theme_bw(base_size = 11) +
    ggplot2::theme(
      axis.text.x      = ggplot2::element_text(angle = 30, hjust = 1),
      legend.position  = "bottom",
      panel.grid.minor = ggplot2::element_blank()
    )
}

panels <- lapply(key_feats_present, make_key_panel)
names(panels) <- key_feats_present

if (length(key_feats_present) == 5) {
  combined <- (panels[[1]] + panels[[2]] + panels[[3]]) /
              (panels[[4]] + panels[[5]] + patchwork::plot_spacer()) +
    patchwork::plot_layout(guides = "collect") &
    ggplot2::theme(legend.position = "bottom")
} else {
  combined <- patchwork::wrap_plots(panels, ncol = 3) +
    patchwork::plot_layout(guides = "collect") &
    ggplot2::theme(legend.position = "bottom")
}

ggplot2::ggsave(
  file.path(dir_figures_lsmean, "key_features_lsmean_panel.png"),
  combined, width = 13, height = 9, dpi = 150
)
ggplot2::ggsave(
  file.path(dir_figures_lsmean, "key_features_lsmean_panel.pdf"),
  combined, width = 13, height = 9
)
message("Saved key_features_lsmean_panel.png and .pdf to: ", dir_figures_lsmean)
