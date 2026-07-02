# 08_spaghetti_plots.R
# Spaghetti plots: individual pig-pair trajectories (opaque foreground)
# with group LS mean +/- 95% CI as dashed semi-transparent context.
#
# Visual hierarchy (focus on individual animals):
#   Foreground — pig lines: solid, linewidth 0.7, alpha 0.90; distinct shape per pig
#   Context    — LS mean : dashed, linewidth 1.2, alpha 0.45
#
# Legend: pig names colour-coded by group (ggtext HTML spans);
#         S pigs on row 1, U pigs on row 2 (guide ncol = 4).
#
# Outputs (all to results/figures/Spaghetti/):
#   {feature}_spaghetti.png  — one per feature
#   key_features_spaghetti_panel.png / .pdf  — 5-panel combined figure

`%+%` <- function(a, b) paste0(a, b)

if (!exists("project_root")) {
  source("scripts/00_setup.R")
}
if (!exists("dat_clean")) {
  source("scripts/01_data_audit.R")
}

# ── Load LS means ──────────────────────────────────────────────────────────────
lsm_spag <- if (exists("lmm_lsmeans_all", envir = .GlobalEnv) &&
                !is.null(get("lmm_lsmeans_all", envir = .GlobalEnv))) {
  get("lmm_lsmeans_all", envir = .GlobalEnv)
} else {
  readxl::read_excel(
    file.path(dir_tables, "lmm_lsmeans_stage_treatment_all_features.xlsx")
  )
}
if ("lower.CL" %in% names(lsm_spag))
  lsm_spag <- dplyr::rename(lsm_spag, lower_95_CI = lower.CL, upper_95_CI = upper.CL)
if ("emmean" %in% names(lsm_spag))
  lsm_spag <- dplyr::rename(lsm_spag, ls_mean = emmean)
lsm_spag <- lsm_spag %>%
  dplyr::mutate(Stage = factor(Stage, levels = stage_levels))

# ── Pig display names and group assignments ────────────────────────────────────
pig_name_map <- c(
  "Elvis"              = "Pig 1 (M)",
  "Cher and Adele"     = "Pig 8&9 (F)",
  "Barry and Stevie"   = "Pig 16&17 (M)",
  "Snoop and Dre"      = "Pig 18&19 (M)",
  "Michael and Prince" = "Pig 10&11 (M)",
  "Paul and John"      = "Pig 14&15 (M)",
  "Tina and Aretha"    = "Pig 20&21 (F)"
)

# S group first (fills row 1 with ncol=4), U group second (row 2)
s_pigs <- c("Pig 1 (M)", "Pig 8&9 (F)", "Pig 16&17 (M)", "Pig 18&19 (M)")
u_pigs <- c("Pig 10&11 (M)", "Pig 14&15 (M)", "Pig 20&21 (F)")
pig_order <- c(s_pigs, u_pigs)

# ── Visual constants ───────────────────────────────────────────────────────────
pal   <- c(S = "#1b7837", U = "#c0392b")
dodge <- ggplot2::position_dodge(width = 0.20)

# Distinct open/hollow shapes — one per pig, traceable across stages
subj_shapes <- c(
  "Pig 1 (M)"      =  1,   # circle
  "Pig 8&9 (F)"    =  2,   # triangle up
  "Pig 16&17 (M)"  =  0,   # square
  "Pig 18&19 (M)"  =  5,   # diamond
  "Pig 10&11 (M)"  =  6,   # triangle down
  "Pig 14&15 (M)"  =  8,   # star
  "Pig 20&21 (F)"  = 10    # circle-plus
)

# HTML-coloured legend labels (rendered by ggtext::element_markdown)
# "&" → "&amp;" so HTML inside <span> is valid
html_safe <- function(x) gsub("&", "&amp;", x, fixed = TRUE)
shape_labels_html <- setNames(
  ifelse(
    pig_order %in% s_pigs,
    sprintf('<span style="color:%s">%s</span>', pal["S"], html_safe(pig_order)),
    sprintf('<span style="color:%s">%s</span>', pal["U"], html_safe(pig_order))
  ),
  pig_order
)

# Symbol colours for legend override (S = green, U = red)
legend_sym_colours <- c(rep(unname(pal["S"]), length(s_pigs)),
                        rep(unname(pal["U"]), length(u_pigs)))

# ── Pre-compute pig-pair stage means (all features) ───────────────────────────
pig_means_all <- purrr::map_dfr(feature_cols, function(feat) {
  dat_clean %>%
    dplyr::filter(!is.na(.data[[feat]])) %>%
    dplyr::group_by(Subject, Treatment, Stage) %>%
    dplyr::summarise(pig_mean = mean(.data[[feat]], na.rm = TRUE), .groups = "drop") %>%
    dplyr::mutate(Stage = factor(Stage, levels = stage_levels), feature = feat)
}) %>%
  dplyr::mutate(
    Subject = factor(
      dplyr::case_match(
        as.character(Subject),
        "Elvis"              ~ "Pig 1 (M)",
        "Cher and Adele"     ~ "Pig 8&9 (F)",
        "Barry and Stevie"   ~ "Pig 16&17 (M)",
        "Snoop and Dre"      ~ "Pig 18&19 (M)",
        "Michael and Prince" ~ "Pig 10&11 (M)",
        "Paul and John"      ~ "Pig 14&15 (M)",
        "Tina and Aretha"    ~ "Pig 20&21 (F)",
        .default = as.character(Subject)
      ),
      levels = pig_order
    )
  )

# ── Helper: build one spaghetti plot ──────────────────────────────────────────
make_spag <- function(feat, pig_df, lsm_df) {
  pig_sub <- pig_df %>% dplyr::filter(feature == feat, !is.na(pig_mean))
  lsm_sub <- lsm_df %>% dplyr::filter(feature == feat)
  if (nrow(pig_sub) == 0) return(NULL)

  p <- ggplot2::ggplot() +

    # ── Foreground: individual pig-pair lines (solid, opaque) ─────────────────
    ggplot2::geom_line(
      data = pig_sub,
      ggplot2::aes(x = Stage, y = pig_mean, colour = Treatment, group = Subject),
      linewidth = 0.7, alpha = 0.90, linetype = "solid", position = dodge
    ) +
    ggplot2::geom_point(
      data = pig_sub,
      ggplot2::aes(x = Stage, y = pig_mean, colour = Treatment,
                   shape = Subject, group = Subject),
      size = 2.5, alpha = 0.95, position = dodge
    )

  if (nrow(lsm_sub) > 0) {
    p <- p +
      # ── Context: group LS mean +/- 95% CI (dashed, semi-transparent) ────────
      ggplot2::geom_line(
        data = lsm_sub,
        ggplot2::aes(x = Stage, y = ls_mean, colour = Treatment, group = Treatment),
        linewidth = 1.2, alpha = 0.40, linetype = "dashed", position = dodge
      ) +
      ggplot2::geom_errorbar(
        data = lsm_sub,
        ggplot2::aes(x = Stage, ymin = lower_95_CI, ymax = upper_95_CI,
                     colour = Treatment, group = Treatment),
        width = 0.12, linewidth = 0.8, alpha = 0.40, position = dodge
      ) +
      ggplot2::geom_point(
        data = lsm_sub,
        ggplot2::aes(x = Stage, y = ls_mean, colour = Treatment, group = Treatment),
        shape = 18, size = 3.5, alpha = 0.40, position = dodge
      )
  }

  p +
    ggplot2::scale_colour_manual(
      values = pal, name = "Treatment",
      guide  = ggplot2::guide_legend(
        order = 1,
        override.aes = list(linewidth = 1.2, shape = NA, alpha = 1)
      )
    ) +
    ggplot2::scale_shape_manual(
      values = subj_shapes,
      labels = shape_labels_html,
      name   = "Subject — points = pig-pair mean per stage",
      guide  = ggplot2::guide_legend(
        order = 2, ncol = 4,
        override.aes = list(
          colour   = legend_sym_colours,
          alpha    = 1, size = 3.5, linetype = 0
        )
      )
    ) +
    ggplot2::labs(
      title   = feat,
      x       = "Stage", y = "Value",
      caption = paste0(
        "Points/solid lines: mean of all squeals per pig-pair at each stage. ",
        "Dashed: Group LS mean ± 95% CI (LMM)."
      )
    ) +
    ggplot2::theme_bw(base_size = 11) +
    ggplot2::theme(
      plot.title        = ggplot2::element_text(face = "bold", size = 11),
      plot.caption      = ggplot2::element_text(size = 7, colour = "grey50",
                                                 hjust = 0),
      legend.position   = "bottom",
      legend.box        = "vertical",
      legend.text       = ggtext::element_markdown(size = 9),
      legend.title      = ggplot2::element_text(size = 9, face = "bold"),
      legend.key.size   = ggplot2::unit(0.85, "lines"),
      legend.spacing.y  = ggplot2::unit(1, "mm"),
      legend.margin     = ggplot2::margin(t = 2, b = 2, unit = "mm"),
      plot.margin       = ggplot2::margin(t = 5, r = 12, b = 5, l = 5,
                                          unit = "mm"),
      panel.grid.minor  = ggplot2::element_blank()
    )
}

# ── Per-feature spaghetti plots ────────────────────────────────────────────────
n_plots <- 0
for (feat in feature_cols) {
  tryCatch({
    p <- make_spag(feat, pig_means_all, lsm_spag)
    if (!is.null(p)) {
      ggplot2::ggsave(
        file.path(dir_figures_spaghetti, safe_name(feat) %+% "_spaghetti.png"),
        p, width = 8, height = 6.5, dpi = 150
      )
      n_plots <- n_plots + 1
    }
  }, error = function(e) {
    log_msg("Spaghetti failed: " %+% feat %+% ": " %+% conditionMessage(e))
  })
}

# ── Combined 5-panel figure: key features ─────────────────────────────────────
key_feats_spag <- intersect(
  c("gfcc_4_std", "pncc_4_std", "s_entropy_mean", "s_flux_mean", "Schlegel21_HNR"),
  feature_cols
)

key_feat_labels <- c(
  gfcc_4_std     = "GFCC 4 (std)",
  pncc_4_std     = "PNCC 4 (std)",
  s_entropy_mean = "Spectral Entropy (mean)",
  s_flux_mean    = "Spectral Flux (mean)",
  Schlegel21_HNR = "HNR (Schlegel21)"
)

if (length(key_feats_spag) >= 2) {
  pig_panel <- pig_means_all %>%
    dplyr::filter(feature %in% key_feats_spag) %>%
    dplyr::mutate(feature = factor(feature, levels = key_feats_spag))

  lsm_panel <- lsm_spag %>%
    dplyr::filter(feature %in% key_feats_spag) %>%
    dplyr::mutate(feature = factor(feature, levels = key_feats_spag))

  p_panel <- ggplot2::ggplot() +
    ggplot2::geom_line(
      data = pig_panel,
      ggplot2::aes(x = Stage, y = pig_mean, colour = Treatment, group = Subject),
      linewidth = 0.7, alpha = 0.90, linetype = "solid", position = dodge
    ) +
    ggplot2::geom_point(
      data = pig_panel,
      ggplot2::aes(x = Stage, y = pig_mean, colour = Treatment,
                   shape = Subject, group = Subject),
      size = 2.2, alpha = 0.95, position = dodge
    ) +
    ggplot2::geom_line(
      data = lsm_panel,
      ggplot2::aes(x = Stage, y = ls_mean, colour = Treatment, group = Treatment),
      linewidth = 1.2, alpha = 0.40, linetype = "dashed", position = dodge
    ) +
    ggplot2::geom_errorbar(
      data = lsm_panel,
      ggplot2::aes(x = Stage, ymin = lower_95_CI, ymax = upper_95_CI,
                   colour = Treatment, group = Treatment),
      width = 0.12, linewidth = 0.8, alpha = 0.40, position = dodge
    ) +
    ggplot2::geom_point(
      data = lsm_panel,
      ggplot2::aes(x = Stage, y = ls_mean, colour = Treatment, group = Treatment),
      shape = 18, size = 3.2, alpha = 0.40, position = dodge
    ) +
    ggplot2::facet_wrap(
      ~ feature, ncol = 1, scales = "free_y",
      labeller = ggplot2::labeller(feature = key_feat_labels)
    ) +
    ggplot2::scale_colour_manual(
      values = pal, name = "Treatment",
      guide  = ggplot2::guide_legend(
        order = 1,
        override.aes = list(linewidth = 1.2, shape = NA, alpha = 1)
      )
    ) +
    ggplot2::scale_shape_manual(
      values = subj_shapes,
      labels = shape_labels_html,
      name   = "Subject — points = pig-pair mean per stage",
      guide  = ggplot2::guide_legend(
        order = 2, ncol = 4,
        override.aes = list(
          colour   = legend_sym_colours,
          alpha    = 1, size = 3.2, linetype = 0
        )
      )
    ) +
    ggplot2::labs(
      title   = "Key Feature Trajectories — Individual Pig Pairs",
      x       = "Stage", y = "Value",
      caption = paste0(
        "Points/solid lines: mean of all squeals per pig-pair at each stage. ",
        "Dashed: Group LS mean ± 95% CI (LMM)."
      )
    ) +
    ggplot2::theme_bw(base_size = 11) +
    ggplot2::theme(
      strip.text        = ggplot2::element_text(face = "bold", size = 9),
      plot.title        = ggplot2::element_text(face = "bold", size = 11),
      plot.caption      = ggplot2::element_text(size = 7, colour = "grey50",
                                                 hjust = 0),
      axis.text.x       = ggplot2::element_text(angle = 30, hjust = 1),
      legend.position   = "bottom",
      legend.box        = "vertical",
      legend.text       = ggtext::element_markdown(size = 9),
      legend.title      = ggplot2::element_text(size = 9, face = "bold"),
      legend.key.size   = ggplot2::unit(0.85, "lines"),
      legend.spacing.y  = ggplot2::unit(1, "mm"),
      legend.margin     = ggplot2::margin(t = 2, b = 2, unit = "mm"),
      plot.margin       = ggplot2::margin(t = 5, r = 12, b = 5, l = 5,
                                          unit = "mm"),
      panel.grid.minor  = ggplot2::element_blank()
    )

  panel_h <- 3.5 * length(key_feats_spag) + 4.0

  ggplot2::ggsave(
    file.path(dir_figures_spaghetti, "key_features_spaghetti_panel.png"),
    p_panel, width = 8, height = panel_h, dpi = 150
  )
  ggplot2::ggsave(
    file.path(dir_figures_spaghetti, "key_features_spaghetti_panel.pdf"),
    p_panel, width = 8, height = panel_h
  )
  log_msg("Saved: key_features_spaghetti_panel.png/pdf")
}

message("08_spaghetti_plots.R complete — ", n_plots,
        " spaghetti plots saved to: ", dir_figures_spaghetti)
