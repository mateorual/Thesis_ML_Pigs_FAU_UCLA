# Feature Selection Summary — Groups S, U, B

## 1. Candidate counts

| Group | Strong | Moderate | Total → LMM |
|-------|--------|----------|-------------|
| S | 16 | 2 | 18 |
| U | 27 | 10 | 37 |
| B | 12 | 0 | 12 |

## 2. Selected features per group
### Group S
| feature                | family     |   mean_pi_uncapped | alpha_consistency_label   | final_decision     |
|:-----------------------|:-----------|-------------------:|:--------------------------|:-------------------|
| lpcc_9_mean            | LPC        |             1      | High                      | Strong candidate   |
| gfcc_8_mean            | GFCC       |             1      | High                      | Strong candidate   |
| gfcc_0_mean            | GFCC       |             1      | High                      | Strong candidate   |
| s_flux_mean            | Spectral   |             1      | High                      | Strong candidate   |
| s_contrast_band_7_mean | Spectral   |             1      | High                      | Strong candidate   |
| lsf_4_std              | LPC        |             1      | High                      | Strong candidate   |
| lpcc_5_mean            | LPC        |             1      | High                      | Strong candidate   |
| mfcc_2_mean            | MFCC       |             0.999  | High                      | Strong candidate   |
| Schlegel21_HNR         | Schlegel21 |             0.988  | High                      | Strong candidate   |
| gfcc_6_mean            | GFCC       |             0.902  | High                      | Strong candidate   |
| lpcc_10_mean           | LPC        |             0.8015 | High                      | Strong candidate   |
| s_contrast_band_5_mean | Spectral   |             0.7965 | High                      | Strong candidate   |
| mfcc_7_mean            | MFCC       |             0.767  | High                      | Strong candidate   |
| s_entropy_mean         | Spectral   |             0.756  | High                      | Strong candidate   |
| pncc_6_mean            | PNCC       |             0.6295 | High                      | Strong candidate   |
| ddgfcc_0_std           | GFCC       |             0.5605 | Moderate                  | Moderate candidate |
| lpcc_4_mean            | LPC        |             0.5185 | High                      | Strong candidate   |
| mfcc_6_mean            | MFCC       |             0.4165 | Moderate                  | Moderate candidate |

### Group U
| feature                | family     |   mean_pi_uncapped | alpha_consistency_label   | final_decision     |
|:-----------------------|:-----------|-------------------:|:--------------------------|:-------------------|
| s_contrast_band_1_mean | Spectral   |             1      | High                      | Strong candidate   |
| ucla_flux              | UCLA       |             1      | High                      | Strong candidate   |
| gfcc_8_mean            | GFCC       |             0.9995 | High                      | Strong candidate   |
| lpcc_10_mean           | LPC        |             0.9995 | High                      | Strong candidate   |
| lpcc_6_mean            | LPC        |             0.999  | High                      | Strong candidate   |
| gfcc_7_mean            | GFCC       |             0.993  | High                      | Strong candidate   |
| mfcc_6_mean            | MFCC       |             0.9855 | High                      | Strong candidate   |
| lpcc_2_mean            | LPC        |             0.9755 | High                      | Strong candidate   |
| e_std                  | Temporal   |             0.97   | High                      | Strong candidate   |
| lsf_8_std              | LPC        |             0.9545 | High                      | Strong candidate   |
| jit_factor             | GAT        |             0.937  | High                      | Strong candidate   |
| s_logdist_std          | Spectral   |             0.936  | High                      | Strong candidate   |
| pncc_4_std             | PNCC       |             0.8745 | High                      | Strong candidate   |
| dpncc_1_std            | PNCC       |             0.8715 | High                      | Strong candidate   |
| lsf_3_mean             | LPC        |             0.839  | High                      | Strong candidate   |
| dpncc_8_std            | PNCC       |             0.8205 | High                      | Strong candidate   |
| mfcc_2_std             | MFCC       |             0.8185 | High                      | Strong candidate   |
| ddmfcc_7_std           | MFCC       |             0.804  | High                      | Strong candidate   |
| lpc_11_mean            | LPC        |             0.793  | High                      | Strong candidate   |
| Schlegel21_PF          | Schlegel21 |             0.775  | High                      | Strong candidate   |
| plpcc_1_std            | PLPCC      |             0.748  | High                      | Strong candidate   |
| lpcc_7_std             | LPC        |             0.741  | High                      | Strong candidate   |
| lpcc_8_mean            | LPC        |             0.723  | High                      | Strong candidate   |
| gfcc_7_std             | GFCC       |             0.7105 | High                      | Strong candidate   |
| gfcc_1_std             | GFCC       |             0.7095 | High                      | Strong candidate   |
| ppq5                   | GAT        |             0.697  | High                      | Strong candidate   |
| pncc_9_std             | PNCC       |             0.6595 | Moderate                  | Moderate candidate |
| plpcc_4_std            | PLPCC      |             0.658  | Moderate                  | Moderate candidate |
| lpc_4_mean             | LPC        |             0.6145 | Moderate                  | Moderate candidate |
| mfcc_11_mean           | MFCC       |             0.611  | High                      | Moderate candidate |
| zcr_std                | Temporal   |             0.5845 | Moderate                  | Moderate candidate |
| plpcc_12_mean          | PLPCC      |             0.582  | Moderate                  | Moderate candidate |
| ddgfcc_0_std           | GFCC       |             0.553  | Moderate                  | Moderate candidate |
| plpcc_10_mean          | PLPCC      |             0.542  | High                      | Strong candidate   |
| gfcc_4_std             | GFCC       |             0.5085 | Moderate                  | Moderate candidate |
| pncc_7_std             | PNCC       |             0.482  | Moderate                  | Moderate candidate |
| s_contrast_band_7_mean | Spectral   |             0.398  | Moderate                  | Moderate candidate |

### Group B
| feature       | family   |   mean_pi_uncapped | alpha_consistency_label   | final_decision   |
|:--------------|:---------|-------------------:|:--------------------------|:-----------------|
| lsf_5_mean    | LPC      |             1      | High                      | Strong candidate |
| mfcc_1_mean   | MFCC     |             1      | High                      | Strong candidate |
| plpcc_11_mean | PLPCC    |             0.9995 | High                      | Strong candidate |
| gfcc_10_mean  | GFCC     |             0.9895 | High                      | Strong candidate |
| mfcc_7_mean   | MFCC     |             0.986  | High                      | Strong candidate |
| lpcc_8_mean   | LPC      |             0.9425 | High                      | Strong candidate |
| lpcc_9_mean   | LPC      |             0.9195 | High                      | Strong candidate |
| lpcc_9_std    | LPC      |             0.839  | High                      | Strong candidate |
| lsf_12_std    | LPC      |             0.8285 | High                      | Strong candidate |
| plpcc_7_mean  | PLPCC    |             0.8205 | High                      | Strong candidate |
| lpc_3_mean    | LPC      |             0.758  | High                      | Strong candidate |
| lpcc_2_mean   | LPC      |             0.684  | High                      | Strong candidate |

## 3. Intersections

### S ∩ U ∩ B  (0 features)
_None_

### S ∩ U only  (5 features, excl. B)
| feature                | family   |   mean_pi_S | decision_S         |   mean_pi_U | decision_U         |
|:-----------------------|:---------|------------:|:-------------------|------------:|:-------------------|
| ddgfcc_0_std           | GFCC     |      0.5605 | Moderate candidate |      0.553  | Moderate candidate |
| gfcc_8_mean            | GFCC     |      1      | Strong candidate   |      0.9995 | Strong candidate   |
| lpcc_10_mean           | LPC      |      0.8015 | Strong candidate   |      0.9995 | Strong candidate   |
| mfcc_6_mean            | MFCC     |      0.4165 | Moderate candidate |      0.9855 | Strong candidate   |
| s_contrast_band_7_mean | Spectral |      1      | Strong candidate   |      0.398  | Moderate candidate |

### U ∩ B only  (2 features, excl. S)
| feature     | family   |   mean_pi_U | decision_U       |   mean_pi_B | decision_B       |
|:------------|:---------|------------:|:-----------------|------------:|:-----------------|
| lpcc_2_mean | LPC      |      0.9755 | Strong candidate |      0.684  | Strong candidate |
| lpcc_8_mean | LPC      |      0.723  | Strong candidate |      0.9425 | Strong candidate |

### S ∩ B only  (2 features, excl. U)
| feature     | family   |   mean_pi_S | decision_S       |   mean_pi_B | decision_B       |
|:------------|:---------|------------:|:-----------------|------------:|:-----------------|
| lpcc_9_mean | LPC      |       1     | Strong candidate |      0.9195 | Strong candidate |
| mfcc_7_mean | MFCC     |       0.767 | Strong candidate |      0.986  | Strong candidate |

## 4. Unions

### S ∪ U  (50 features — selected in Scar OR Unilateral)

A `True` in the `selected_*` columns indicates the feature was selected for that group.

| feature                | family     | mean_pi_S   | selected_S   | mean_pi_U   | selected_U   |
|:-----------------------|:-----------|:------------|:-------------|:------------|:-------------|
| Schlegel21_HNR         | Schlegel21 | 0.988       | True         |             | False        |
| Schlegel21_PF          | Schlegel21 |             | False        | 0.775       | True         |
| ddgfcc_0_std           | GFCC       | 0.5605      | True         | 0.553       | True         |
| ddmfcc_7_std           | MFCC       |             | False        | 0.804       | True         |
| dpncc_1_std            | PNCC       |             | False        | 0.8715      | True         |
| dpncc_8_std            | PNCC       |             | False        | 0.8205      | True         |
| e_std                  | Temporal   |             | False        | 0.97        | True         |
| gfcc_0_mean            | GFCC       | 1.0         | True         |             | False        |
| gfcc_1_std             | GFCC       |             | False        | 0.7095      | True         |
| gfcc_4_std             | GFCC       |             | False        | 0.5085      | True         |
| gfcc_6_mean            | GFCC       | 0.902       | True         |             | False        |
| gfcc_7_mean            | GFCC       |             | False        | 0.993       | True         |
| gfcc_7_std             | GFCC       |             | False        | 0.7105      | True         |
| gfcc_8_mean            | GFCC       | 1.0         | True         | 0.9995      | True         |
| jit_factor             | GAT        |             | False        | 0.937       | True         |
| lpc_11_mean            | LPC        |             | False        | 0.793       | True         |
| lpc_4_mean             | LPC        |             | False        | 0.6145      | True         |
| lpcc_10_mean           | LPC        | 0.8015      | True         | 0.9995      | True         |
| lpcc_2_mean            | LPC        |             | False        | 0.9755      | True         |
| lpcc_4_mean            | LPC        | 0.5185      | True         |             | False        |
| lpcc_5_mean            | LPC        | 1.0         | True         |             | False        |
| lpcc_6_mean            | LPC        |             | False        | 0.999       | True         |
| lpcc_7_std             | LPC        |             | False        | 0.741       | True         |
| lpcc_8_mean            | LPC        |             | False        | 0.723       | True         |
| lpcc_9_mean            | LPC        | 1.0         | True         |             | False        |
| lsf_3_mean             | LPC        |             | False        | 0.839       | True         |
| lsf_4_std              | LPC        | 1.0         | True         |             | False        |
| lsf_8_std              | LPC        |             | False        | 0.9545      | True         |
| mfcc_11_mean           | MFCC       |             | False        | 0.611       | True         |
| mfcc_2_mean            | MFCC       | 0.999       | True         |             | False        |
| mfcc_2_std             | MFCC       |             | False        | 0.8185      | True         |
| mfcc_6_mean            | MFCC       | 0.4165      | True         | 0.9855      | True         |
| mfcc_7_mean            | MFCC       | 0.767       | True         |             | False        |
| plpcc_10_mean          | PLPCC      |             | False        | 0.542       | True         |
| plpcc_12_mean          | PLPCC      |             | False        | 0.582       | True         |
| plpcc_1_std            | PLPCC      |             | False        | 0.748       | True         |
| plpcc_4_std            | PLPCC      |             | False        | 0.658       | True         |
| pncc_4_std             | PNCC       |             | False        | 0.8745      | True         |
| pncc_6_mean            | PNCC       | 0.6295      | True         |             | False        |
| pncc_7_std             | PNCC       |             | False        | 0.482       | True         |
| pncc_9_std             | PNCC       |             | False        | 0.6595      | True         |
| ppq5                   | GAT        |             | False        | 0.697       | True         |
| s_contrast_band_1_mean | Spectral   |             | False        | 1.0         | True         |
| s_contrast_band_5_mean | Spectral   | 0.7965      | True         |             | False        |
| s_contrast_band_7_mean | Spectral   | 1.0         | True         | 0.398       | True         |
| s_entropy_mean         | Spectral   | 0.756       | True         |             | False        |
| s_flux_mean            | Spectral   | 1.0         | True         |             | False        |
| s_logdist_std          | Spectral   |             | False        | 0.936       | True         |
| ucla_flux              | UCLA       |             | False        | 1.0         | True         |
| zcr_std                | Temporal   |             | False        | 0.5845      | True         |

### S ∪ U ∪ B  (58 features — selected in any group)

| feature                | family     | mean_pi_S   | selected_S   | mean_pi_U   | selected_U   | mean_pi_B   | selected_B   |
|:-----------------------|:-----------|:------------|:-------------|:------------|:-------------|:------------|:-------------|
| Schlegel21_HNR         | Schlegel21 | 0.988       | True         |             | False        |             | False        |
| Schlegel21_PF          | Schlegel21 |             | False        | 0.775       | True         |             | False        |
| ddgfcc_0_std           | GFCC       | 0.5605      | True         | 0.553       | True         |             | False        |
| ddmfcc_7_std           | MFCC       |             | False        | 0.804       | True         |             | False        |
| dpncc_1_std            | PNCC       |             | False        | 0.8715      | True         |             | False        |
| dpncc_8_std            | PNCC       |             | False        | 0.8205      | True         |             | False        |
| e_std                  | Temporal   |             | False        | 0.97        | True         |             | False        |
| gfcc_0_mean            | GFCC       | 1.0         | True         |             | False        |             | False        |
| gfcc_10_mean           | GFCC       |             | False        |             | False        | 0.9895      | True         |
| gfcc_1_std             | GFCC       |             | False        | 0.7095      | True         |             | False        |
| gfcc_4_std             | GFCC       |             | False        | 0.5085      | True         |             | False        |
| gfcc_6_mean            | GFCC       | 0.902       | True         |             | False        |             | False        |
| gfcc_7_mean            | GFCC       |             | False        | 0.993       | True         |             | False        |
| gfcc_7_std             | GFCC       |             | False        | 0.7105      | True         |             | False        |
| gfcc_8_mean            | GFCC       | 1.0         | True         | 0.9995      | True         |             | False        |
| jit_factor             | GAT        |             | False        | 0.937       | True         |             | False        |
| lpc_11_mean            | LPC        |             | False        | 0.793       | True         |             | False        |
| lpc_3_mean             | LPC        |             | False        |             | False        | 0.758       | True         |
| lpc_4_mean             | LPC        |             | False        | 0.6145      | True         |             | False        |
| lpcc_10_mean           | LPC        | 0.8015      | True         | 0.9995      | True         |             | False        |
| lpcc_2_mean            | LPC        |             | False        | 0.9755      | True         | 0.684       | True         |
| lpcc_4_mean            | LPC        | 0.5185      | True         |             | False        |             | False        |
| lpcc_5_mean            | LPC        | 1.0         | True         |             | False        |             | False        |
| lpcc_6_mean            | LPC        |             | False        | 0.999       | True         |             | False        |
| lpcc_7_std             | LPC        |             | False        | 0.741       | True         |             | False        |
| lpcc_8_mean            | LPC        |             | False        | 0.723       | True         | 0.9425      | True         |
| lpcc_9_mean            | LPC        | 1.0         | True         |             | False        | 0.9195      | True         |
| lpcc_9_std             | LPC        |             | False        |             | False        | 0.839       | True         |
| lsf_12_std             | LPC        |             | False        |             | False        | 0.8285      | True         |
| lsf_3_mean             | LPC        |             | False        | 0.839       | True         |             | False        |
| lsf_4_std              | LPC        | 1.0         | True         |             | False        |             | False        |
| lsf_5_mean             | LPC        |             | False        |             | False        | 1.0         | True         |
| lsf_8_std              | LPC        |             | False        | 0.9545      | True         |             | False        |
| mfcc_11_mean           | MFCC       |             | False        | 0.611       | True         |             | False        |
| mfcc_1_mean            | MFCC       |             | False        |             | False        | 1.0         | True         |
| mfcc_2_mean            | MFCC       | 0.999       | True         |             | False        |             | False        |
| mfcc_2_std             | MFCC       |             | False        | 0.8185      | True         |             | False        |
| mfcc_6_mean            | MFCC       | 0.4165      | True         | 0.9855      | True         |             | False        |
| mfcc_7_mean            | MFCC       | 0.767       | True         |             | False        | 0.986       | True         |
| plpcc_10_mean          | PLPCC      |             | False        | 0.542       | True         |             | False        |
| plpcc_11_mean          | PLPCC      |             | False        |             | False        | 0.9995      | True         |
| plpcc_12_mean          | PLPCC      |             | False        | 0.582       | True         |             | False        |
| plpcc_1_std            | PLPCC      |             | False        | 0.748       | True         |             | False        |
| plpcc_4_std            | PLPCC      |             | False        | 0.658       | True         |             | False        |
| plpcc_7_mean           | PLPCC      |             | False        |             | False        | 0.8205      | True         |
| pncc_4_std             | PNCC       |             | False        | 0.8745      | True         |             | False        |
| pncc_6_mean            | PNCC       | 0.6295      | True         |             | False        |             | False        |
| pncc_7_std             | PNCC       |             | False        | 0.482       | True         |             | False        |
| pncc_9_std             | PNCC       |             | False        | 0.6595      | True         |             | False        |
| ppq5                   | GAT        |             | False        | 0.697       | True         |             | False        |
| s_contrast_band_1_mean | Spectral   |             | False        | 1.0         | True         |             | False        |
| s_contrast_band_5_mean | Spectral   | 0.7965      | True         |             | False        |             | False        |
| s_contrast_band_7_mean | Spectral   | 1.0         | True         | 0.398       | True         |             | False        |
| s_entropy_mean         | Spectral   | 0.756       | True         |             | False        |             | False        |
| s_flux_mean            | Spectral   | 1.0         | True         |             | False        |             | False        |
| s_logdist_std          | Spectral   |             | False        | 0.936       | True         |             | False        |
| ucla_flux              | UCLA       |             | False        | 1.0         | True         |             | False        |
| zcr_std                | Temporal   |             | False        | 0.5845      | True         |             | False        |

## 5. Figures
- `venn_diagram.png` — 3-set Venn diagram of feature overlap
- `family_breakdown.png` — stacked bar of feature families per group