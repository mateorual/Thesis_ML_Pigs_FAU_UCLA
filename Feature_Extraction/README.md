# Feature Extraction

Acoustic feature-extraction pipeline for porcine (pig) vocalization recordings, developed for the
master's thesis *"Machine Learning-Based Recognition and Differentiation of Porcine Voice Acoustics
for Longitudinal Assessment of Laryngeal Surgical Outcomes."*

Given short WAV recordings of pig squeals and their glottal-cycle annotations, the pipeline computes
474 acoustic features per squeal — spanning cycle-based perturbation/noise measures, spectral and
cepstral descriptors (MFCC, GFCC, PLPCC, PNCC), linear-prediction coefficients, spectral energy
ratios, and porcine-specific parameters adapted from prior literature — and consolidates them into a
single tabular dataset for downstream feature selection and statistical modeling.

## Project Structure

```
Feature_Extraction/
├── README.md
├── requirements.txt
├── scripts/
│   ├── main_pig_directory_scan_final.py   # Entry point: scans Audio_Snips/, extracts features;
│   │                                       #   also directly implements the UCLA (5 features) and
│   │                                       #   Schlegel21 (7 features) parameter families below
│   ├── consolidate_features.py            # Merges per-squeal features into one Excel file
│   ├── a_gat.py                           # Cycle-based perturbation & noise (37 features)
│   ├── b_dfg.py                           # Cepstral peak prominence (8 features)
│   ├── c_temporal.py                      # Temporal cycle descriptors (6 features)
│   ├── d_spectral.py                      # STFT spectral shape (32 features)
│   ├── e_cepstral.py                      # MFCC (72) / GFCC (72) / PLPCC (78) / PNCC (72) features
│   ├── f_lpc.py                           # LPC, LSF, LPCC (86 features)
│   ├── i_ratio.py                         # Spectral energy ratios (4 features)
│   └── svfel/                             # Local signal-processing library used by all extractors
├── Detected_Cycles/                       # Provided: GAT glottal-cycle annotations for the thesis dataset
├── Audio_Snips/                           # Not included — see "Input data" below
└── Consolidated_Features_final.xlsx       # Provided: final consolidated feature matrix (thesis dataset)
```

Feature counts above are the raw number of columns each module writes per squeal (verified against
a sample of the actual extracted output), and sum to 479 across the twelve feature families. The
released `Consolidated_Features_final.xlsx` has 474 feature columns per squeal — five columns are
dropped during consolidation; see [Output](#output) below for exactly which ones and why.

`svfel` is a self-contained local package (framing, spectral transforms, filterbanks, cepstral/LPC/
perturbation primitives) that every extractor module imports. Only the files actually needed by the
seven extractor modules above are included here — a few library modules that exist upstream
(`complexity.py`, `pitch.py`, `dfg/had.py`, `dfg/nyquist.py`, `utils/plot.py`) are not part of the
dependency chain used by this pipeline and were left out to keep the repository minimal.

Two additional feature families from the original research pipeline, `g_avca.py` and `h_imf.py`,
require a MATLAB engine and MATLAB toolboxes. They are disabled by default in
`main_pig_directory_scan_final.py` (`EXTRACT_MATLAB_FEATURES = False`) and were **not** used to
produce `Consolidated_Features_final.xlsx`, so they are not included in this repository.

## Setup

### Prerequisites
- Python 3.11 (the pipeline was developed and tested on this version; 3.9+ should work)

### Installation
```bash
pip install -r requirements.txt
```

## Input Data

### 1. Audio recordings (`Audio_Snips/`) — not included

The pipeline expects short WAV snippets (one per candidate vocalization) organized as:

```
Audio_Snips/{Treatment}/week_{N}_{Subject}_{Type}_Snip/{squeal_name}.wav
```

- `{Treatment}` — a folder name of your choosing per treatment group (the thesis dataset used
  `Bilateral`, `Scar`, `Unilateral`).
- `week_{N}_{Subject}_{Type}_Snip` — recording-session folder, where `{N}` is the week number
  relative to surgery, `{Subject}` is the animal or animal-pair identifier, and `{Type}` is a
  single-letter code (`B`, `S`, or `U`) matching the treatment. Example:
  `week_0_Beck and Kurt_B_Snip`.
- `{squeal_name}.wav` — mono or stereo WAV, any sampling rate (`svfel.core.VowelLoader` asserts
  44100 Hz / 16-bit PCM at load time, so upstream recordings must be prepared at that format).

This folder is not distributed with the repository (raw audio is large and not part of the
released dataset). To run the pipeline on your own recordings, recreate this structure under
`Audio_Snips/` at the repository root.

### 2. Glottal-cycle annotations (`Detected_Cycles/`) — included

Each WAV file requires a corresponding `.cycles` file giving the sample-index boundaries of each
glottal cycle, produced with the **Glottal Analysis Toolkit (GAT)** — a separate, GUI-driven,
manual step that is not part of this repository. GAT's output mirrors the `Audio_Snips/` structure:

```
Detected_Cycles/{Treatment}/week_{N}_{Subject}_{Type}_Snip/{squeal_name}/{squeal_name}@{DDMmmYY}(HH-MM).cycles
```

Each `{squeal_name}/` folder can contain more than one `.cycles` file (e.g. after re-running GAT
with different settings); the extraction script always uses the most recent one, based on the
timestamp encoded in the filename.

This repository **includes** the pre-computed `Detected_Cycles/` used for the thesis dataset
(2,357 files, ~14 MB), so the feature-extraction step below can be reproduced directly against the
matching `Audio_Snips/` recordings without repeating the GAT step. If you bring your own audio,
you will need to run GAT (or an equivalent cycle-detection tool producing the same binary format,
read by `svfel.utils.io.read_cycles`) yourself first.

## How to Run

Run both scripts from the `Feature_Extraction/` repository root (so that the relative paths below
resolve correctly), or use absolute paths — both work.

### Step 1 — Extract features per squeal

```bash
python scripts/main_pig_directory_scan_final.py
```

This recursively scans `Audio_Snips/`, matches each WAV file to its cycles folder in
`Detected_Cycles/`, and writes one pickle file per feature family to:

```
Extracted_Features_Structured/{Treatment}/{Subject}/week_{N}/{squeal_name}/*.pickle
```

The script is idempotent — if a pickle file already exists for a given squeal and feature family,
it is not recomputed, so an interrupted run can simply be restarted. Each of the twelve feature
families is wrapped in its own error handler, so a failure in one family (e.g. an LPC fit that
does not converge on an unusually short squeal) does not abort extraction for the rest.

### Step 2 — Consolidate into a single dataset

```bash
python scripts/consolidate_features.py
```

This reads every squeal's pickle files back out of `Extracted_Features_Structured/`, joins them
with metadata (subject, treatment, week, number of detected cycles), and writes:

```
Consolidated_Features_final.xlsx
```

## Output

`Consolidated_Features_final.xlsx` (verified against the file included in this repository) contains
two sheets:

| Sheet | Contents |
|---|---|
| `Features_Data` | 2,357 rows (one per squeal): metadata columns (`Squeal`, `Subject`, `Treatment`, `Week`, `Num_Cycles`) followed by 474 acoustic feature columns, with headers color-coded by source extractor module. |
| `Parameter_Reference` | Per-extractor listing of every feature column name with a short description. |

`consolidate_features.py` also contains logic to generate a `Summary` sheet (one row per extractor
module with feature counts and references); the sheet is not present in the shipped
`Consolidated_Features_final.xlsx`, so `Features_Data` and `Parameter_Reference` are what you will
actually get from this pipeline as configured.

Two column-dropping rules run automatically during consolidation, together removing 5 of the 479
raw feature columns for the thesis dataset:

- **Four columns dropped unconditionally**, regardless of dataset, due to known systematic quality
  issues identified during data auditing: `f0_max` and `f0_min` (quantization / F0-floor artifacts
  in `a_gat.py`), and `snr1_max` / `snr1_min` (an internal 36 dB ceiling and extreme low outliers in
  the same module's SNR computation).
- **Any column with more than 50% missing values** across the dataset is also dropped. For the
  thesis dataset this additionally removed `nne` (normalized noise energy, also from `a_gat.py`).
  This rule is dataset-dependent: re-running the pipeline on different recordings may drop a
  different (possibly empty) set of columns here, so the final feature count for your own dataset
  is not guaranteed to be exactly 474.

The `Consolidated_Features_final.xlsx` file included in this repository is the exact output used
as the starting point for the feature-selection and statistical-modeling stages of the thesis
(see the `Feature_Selection` and `LMM_Analysis` components of this repository).

## Notes

- Signals passed to `a_gat.py`–`i_ratio.py` are preprocessed (50 Hz high-pass filter, mean-centered,
  max-abs normalized); the porcine-specific `z_ucla_parameters` and `z2_schlegel21_parameters`
  feature families instead operate on the raw WAV file directly.
- Frame-based extractors use 4096-sample Hanning-windowed frames with 2048-sample hop (~93 ms /
  ~46 ms at 44.1 kHz), except where noted otherwise (e.g. `f_lpc.py` operates on a resampled
  16 kHz signal with 1024/512-sample frames).
- A detailed, per-parameter rationale for every tunable value in the pipeline (why each cutoff,
  filter range, and window size was chosen for pig squeals rather than left at library defaults
  for human speech) is documented in the thesis's Supplementary Material A.1.

## Troubleshooting

**`ERROR: Audio folder not found`** — `Audio_Snips/` must exist at the repository root before
running `main_pig_directory_scan_final.py`; see [Input Data](#input-data) above.

**`ERROR: Extracted features folder not found`** — run
`scripts/main_pig_directory_scan_final.py` before `scripts/consolidate_features.py`.

**`No cycles folder found for: <squeal>`** — the WAV file has no matching folder under
`Detected_Cycles/`; check that the folder names match exactly, including spacing and the
`week_{N}_{Subject}_{Type}_Snip` pattern.

**`assert self.fs == 44100` / `assert self.bits == 16`** — `svfel.core.VowelLoader` requires
44.1 kHz, 16-bit PCM WAV input; resample/re-encode source recordings if they use a different
format.
