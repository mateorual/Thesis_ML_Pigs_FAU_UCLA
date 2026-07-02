# Statistical Methods

Acoustic parameter values were analyzed at the squeal level (1486 squeal events across 7 pig pairs).

Time was categorized into four stages: Pre, Early Post, Mid Post, and Late Post, reflecting the recording schedule. Stages were treated as an unordered categorical factor because recordings were sparse and irregularly spaced across pigs and weeks.

For each of the 50 acoustic features, a linear mixed-effects model was fit with Stage, Treatment (S vs U), and their interaction (Stage × Treatment) as fixed effects, and pig pair (Subject) as a random intercept to account for within-pair correlation arising from repeated measurements. Models were fit using the lme4/lmerTest packages in R (REML estimation; Satterthwaite degrees-of-freedom approximation). Of 50 features: 50 were fit with the full Stage × Treatment interaction model; 0 required an additive Stage + Treatment fallback (interaction term could not be estimated); and 0 models failed entirely.

Raw means were computed for direct interpretability. Least-squares (LS) means and 95% confidence intervals were estimated from each model using the emmeans package to support fair comparisons in the unbalanced dataset.

Type III fixed-effect tests (F-statistics with Satterthwaite df) were used for primary inference, with emphasis on the Stage × Treatment interaction p-value as the key test of whether treatment groups differ across recovery stages. Exploratory pairwise post-hoc contrasts (emmeans, Benjamini-Hochberg adjustment) were computed as secondary outputs.

Kruskal-Wallis tests (Treatment effect within each Stage) and Wilcoxon rank-sum tests (Pre vs each post stage within each pig pair) were performed as secondary/descriptive analyses only. These nonparametric tests assume independent observations and do not fully account for repeated measurements within pig pair; they should not be interpreted as primary evidence of treatment effects.

All analyses were performed in R version 4.5.2. Significance level α = 0.05 was used for primary inference.

