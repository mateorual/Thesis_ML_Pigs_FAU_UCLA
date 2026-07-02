# 06_trajectory_plots.R
# Per-feature trajectory plots: raw means ± SE and LS means ± 95% CI
# Both treatment groups (S, U) overlaid across the four time stages.
# Saved to results/figures/<feature>_trajectory.png

`%+%` <- function(a, b) paste0(a, b)

if (!exists("project_root")) {
  source("scripts/00_setup.R")
}
if (!exists("dat_clean")) {
  source("scripts/01_data_audit.R")
}

# Load tables (use in-memory objects if present, otherwise read from disk)
load_tbl <- function(var, path) {
  if (exists(var, envir = .GlobalEnv) && !is.null(get(var, envir = .GlobalEnv)))
    get(var, envir = .GlobalEnv)
  else
    readxl::read_excel(path)
}

raw_all <- load_tbl("lmm_raw_all",
                    file.path(dir_tables, "raw_means_by_stage_treatment_all_features.xlsx"))
lsm_all <- load_tbl("lmm_lsmeans_all",
                    file.path(dir_tables, "lmm_lsmeans_stage_treatment_all_features.xlsx"))

# Standardise column names coming from disk (emmeans uses lower.CL / upper.CL)
if ("lower.CL" %in% names(lsm_all)) lsm_all <- dplyr::rename(lsm_all, lower_95_CI = lower.CL, upper_95_CI = upper.CL)
if ("emmean"   %in% names(lsm_all)) lsm_all <- dplyr::rename(lsm_all, ls_mean = emmean)

# Colour palette: S = teal, U = coral
pal <- c(S = "#1b7837", U = "#c0392b")
# Stage order on x-axis
stage_ord <- factor(stage_levels, levels = stage_levels)

n_plots <- 0

for (feat in feature_cols) {

  # ── Raw means ± SE per Treatment × Stage ──────────────────────────────────
  raw_sub <- raw_all %>%
    dplyr::filter(feature == feat) %>%
    dplyr::mutate(
      Stage     = factor(Stage, levels = stage_levels),
      se        = sd / sqrt(n_squeals),
      ymin_raw  = mean - se,
      ymax_raw  = mean + se
    )

  if (nrow(raw_sub) == 0) next

  # ── LS means ± 95% CI per Treatment × Stage ───────────────────────────────
  lsm_sub <- lsm_all %>%
    dplyr::filter(feature == feat) %>%
    dplyr::mutate(Stage = factor(Stage, levels = stage_levels))

  has_lsm <- nrow(lsm_sub) > 0

  # ── Build plot ─────────────────────────────────────────────────────────────
  # Slight horizontal dodge so error bars don't overlap
  dodge <- ggplot2::position_dodge(width = 0.25)

  p <- ggplot2::ggplot() +

    # ---- Raw means ± SE (filled circles, solid lines) ----------------------
    ggplot2::geom_line(
      data = raw_sub,
      ggplot2::aes(x = Stage, y = mean, colour = Treatment, group = Treatment),
      linewidth = 0.9, linetype = "solid",
      position = dodge
    ) +
    ggplot2::geom_errorbar(
      data = raw_sub,
      ggplot2::aes(x = Stage, ymin = ymin_raw, ymax = ymax_raw,
                   colour = Treatment, group = Treatment),
      width = 0.15, linewidth = 0.7,
      position = dodge
    ) +
    ggplot2::geom_point(
      data = raw_sub,
      ggplot2::aes(x = Stage, y = mean, colour = Treatment,
                   group = Treatment, shape = "Raw mean ± SE"),
      size = 3, position = dodge
    )

  # ---- LS means ± 95% CI (open triangles, dashed lines) -------------------
  if (has_lsm) {
    p <- p +
      ggplot2::geom_line(
        data = lsm_sub,
        ggplot2::aes(x = Stage, y = ls_mean, colour = Treatment, group = Treatment),
        linewidth = 0.8, linetype = "dashed",
        position = dodge
      ) +
      ggplot2::geom_errorbar(
        data = lsm_sub,
        ggplot2::aes(x = Stage, ymin = lower_95_CI, ymax = upper_95_CI,
                     colour = Treatment, group = Treatment),
        width = 0.10, linewidth = 0.6, linetype = "solid",
        position = dodge
      ) +
      ggplot2::geom_point(
        data = lsm_sub,
        ggplot2::aes(x = Stage, y = ls_mean, colour = Treatment,
                     group = Treatment, shape = "LS mean ± 95% CI"),
        size = 3, position = dodge
      )
  }

  p <- p +
    ggplot2::scale_colour_manual(values = pal, name = "Treatment") +
    ggplot2::scale_shape_manual(
      values = c("Raw mean ± SE" = 16, "LS mean ± 95% CI" = 17),
      name = ""
    ) +
    ggplot2::labs(
      title    = feat,
      subtitle = "Solid/circles = raw mean ± SE   |   Dashed/triangles = LS mean ± 95% CI",
      x        = "Stage",
      y        = "Value"
    ) +
    ggplot2::theme_bw(base_size = 11) +
    ggplot2::theme(
      plot.title    = ggplot2::element_text(face = "bold", size = 11),
      plot.subtitle = ggplot2::element_text(size = 8, colour = "grey40"),
      legend.position = "bottom",
      legend.box    = "horizontal",
      panel.grid.minor = ggplot2::element_blank()
    )

  out_path <- file.path(dir_figures, safe_name(feat) %+% "_trajectory.png")
  ggplot2::ggsave(out_path, p, width = 7, height = 4.5, dpi = 150)
  n_plots <- n_plots + 1
}

message("06_trajectory_plots.R complete — ", n_plots, " trajectory plots saved to: ", dir_figures)

# ── LS-mean-only trajectory plots → LS_Mean/ ──────────────────────────────────
# Features that get PDF copies in addition to PNG (key spectral features)
key_feats_ls <- intersect(
  c("gfcc_4_std", "pncc_4_std", "s_entropy_mean", "s_flux_mean", "Schlegel21_HNR"),
  feature_cols
)

n_plots_lsm <- 0

for (feat in feature_cols) {
  lsm_only <- lsm_all %>%
    dplyr::filter(feature == feat) %>%
    dplyr::mutate(Stage = factor(Stage, levels = stage_levels))

  if (nrow(lsm_only) == 0) next

  p_ls <- ggplot2::ggplot(
    lsm_only,
    ggplot2::aes(x = Stage, y = ls_mean, colour = Treatment, group = Treatment)
  ) +
    ggplot2::geom_line(linewidth = 0.9, position = dodge) +
    ggplot2::geom_errorbar(
      ggplot2::aes(ymin = lower_95_CI, ymax = upper_95_CI),
      width = 0.12, linewidth = 0.7, position = dodge
    ) +
    ggplot2::geom_point(size = 3, position = dodge) +
    ggplot2::scale_colour_manual(values = pal, name = "Treatment") +
    ggplot2::labs(
      title    = feat,
      subtitle = "LS mean ± 95% CI  (estimated marginal means from LMM)",
      x = "Stage", y = "Value"
    ) +
    ggplot2::theme_bw(base_size = 11) +
    ggplot2::theme(
      plot.title       = ggplot2::element_text(face = "bold", size = 11),
      plot.subtitle    = ggplot2::element_text(size = 8, colour = "grey40"),
      legend.position  = "bottom",
      panel.grid.minor = ggplot2::element_blank()
    )

  ggplot2::ggsave(
    file.path(dir_figures_lsmean, safe_name(feat) %+% "_trajectory.png"),
    p_ls, width = 7, height = 4.5, dpi = 150
  )

  if (feat %in% key_feats_ls) {
    ggplot2::ggsave(
      file.path(dir_figures_lsmean, safe_name(feat) %+% "_trajectory.pdf"),
      p_ls, width = 7, height = 4.5
    )
  }

  n_plots_lsm <- n_plots_lsm + 1
}

message("  LS-mean-only plots saved to: ", dir_figures_lsmean,
        " (", n_plots_lsm, " PNG + ", length(key_feats_ls), " PDF)")

# ── Combined LS_Mean panel: key features — 3-column × 2-row layout ────────────
if (!requireNamespace("patchwork", quietly = TRUE)) install.packages("patchwork")
library(patchwork)

key_feats_units <- c(
  gfcc_4_std     = "a.u.",
  pncc_4_std     = "a.u.",
  s_entropy_mean = "a.u.",
  s_flux_mean    = "a.u.",
  Schlegel21_HNR = "dB"
)

key_feats_present <- intersect(names(key_feats_units), feature_cols)

if (length(key_feats_present) >= 2) {

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

  # Arrange: 3 in top row, 2 in bottom row (right cell of bottom row left empty)
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
  message("  Combined key-features LS_Mean panel saved.")
}
