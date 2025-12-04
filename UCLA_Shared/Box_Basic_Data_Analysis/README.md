# Box Basic Data Analysis

A comprehensive audio analysis pipeline that accesses audio files from Box cloud storage and performs metadata and fundamental frequency (F0) analysis on pig vocalization recordings.

## Project Structure

```
Box_Basic_Data_Analysis/
├── config.py                    # Configuration and constants
├── main.py                      # Main orchestration script
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── helpers/                     # Box API helpers
│   ├── __init__.py
│   ├── box_auth.py             # Authentication management
│   └── box_client.py           # Box API wrapper
├── analysis/                    # Analysis modules
│   ├── __init__.py
│   ├── metadata_analysis.py    # Metadata extraction and visualization
│   └── f0_analysis.py          # Fundamental frequency analysis
└── output/                      # Generated reports and visualizations
    ├── box_audio_metadata_analysis.xlsx
    ├── box_audio_f0_analysis.xlsx
    └── *.png (visualization files)
```

## Features

### 1. Dual Folder Structure Modes
The analysis supports two different folder organization modes, configurable via `FOLDER_STRUCTURE` in [config.py](Box_Basic_Data_Analysis/config.py):

#### Mode 1: "Individual Pairs DONE"
- Analyzes 8 subject-specific folders (e.g., "P24-1-Elvis-DONE")
- Searches for "Processed" folders within each subject folder (supports nested search)
- Suitable for organized per-subject data with processing stages

#### Mode 2: "All Pig Snips"
- Analyzes centralized "All Pig Snips" folder structure
- Organized by treatment type: Bilateral, Unilateral, Scar
- Processes week-labeled folders (e.g., "week_26_Madonna and Beyonce_B_Snip")
- Automatically extracts subject, week, and treatment from folder names
- Recursively scans nested folders for .wav files

### 2. Box Integration
- OAuth 2.0 authentication with automatic token refresh
- Recursive file scanning within specified folders
- Supports nested folder structures
- Flexible folder pattern matching

### 3. Metadata Analysis
- Audio duration extraction
- File size analysis
- Sampling rate detection
- Week identification from folder paths
- Statistical summaries by subject, week, and treatment type
- Visualizations:
  - Audio count per week by subject
  - Total audio files per subject
  - Duration distribution by subject
  - Total recording duration per subject
  - Subject vs Week heatmap

### 4. F0 (Fundamental Frequency) Analysis
- Autocorrelation-based pitch detection
- Optimized for pig vocalizations (50-500 Hz)
- 30ms snippets with 50% overlap
- Treatment type grouping (Scar, Bilateral, Unilateral)
- Longitudinal analysis (across weeks)
- Statistical summaries by subject, treatment, and week

## Setup

### Prerequisites
- Python 3.7+
- Box Developer account with app credentials

### Installation

1. Install required packages:
```bash
pip install -r requirements.txt
```

2. Get Box credentials:
   - Go to https://app.box.com/developers/console
   - Create or select your application
   - Copy the Client ID and Client Secret
   - Add redirect URI: `http://localhost:3000`

3. Create credentials file:
   Create a file named `box_credentials.txt` in the project root:
   ```
   CLIENT_ID=your_client_id_here
   CLIENT_SECRET=your_client_secret_here
   ```

4. Run the script:
```bash
python main.py
```

On first run, it will guide you through OAuth authentication.

## Configuration

Edit [config.py](Box_Basic_Data_Analysis/config.py) to customize:

### Folder Structure Mode
**`FOLDER_STRUCTURE`**: Select folder organization mode
- `"Individual Pairs DONE"`: Analyze individual subject folders with "Processed" subfolders
- `"All Pig Snips"`: Analyze centralized week-based snip folders

### Analysis Type Selection
**`ANALYSIS_TYPE`**: Choose which analyses to run
- `"Metadata Analysis"`: Only extract and analyze metadata (duration, size, sampling rate, visualizations)
- `"F0 Analysis"`: Only perform fundamental frequency analysis
- `"Metadata and F0 Analysis"`: Run both analyses (default)

### Mode-Specific Settings

**For "Individual Pairs DONE" mode:**
- `MAIN_FOLDERS`: List of 8 subject folders to analyze
- `PROCESSED_FOLDER_PATTERNS`: Patterns to match "Processed" folders

**For "All Pig Snips" mode:**
- `ALL_PIG_SNIPS_FOLDER_ID`: ID of "All Pig Snips" folder (default: '348140860679')
- `ALL_PIG_SNIPS_TREATMENT_FOLDERS`: Mapping of treatment codes to folder names

### General Settings
- `BOX_ROOT_FOLDER_ID`: Root folder ID in Box (default: '237821336437')
- `TREATMENT_MAPPING`: Subject to treatment type mapping (S/B/U)
- `F0_MIN_FREQ`, `F0_MAX_FREQ`: Valid F0 range (default: 50-500 Hz)
- `F0_SNIPPET_DURATION`: Analysis window size (default: 0.03s)
- `F0_OVERLAP`: Overlap between windows (default: 0.5)

## Output Files

### 1. Metadata Excel File
Contains sheets:
- **Audio Files Data**: Complete metadata for all files
- **Summary by Subject**: Aggregated statistics per subject
- **Summary by Week**: Weekly aggregated statistics
- **Overall Summary**: High-level metrics

### 2. F0 Analysis Excel File
Contains sheets:
- **F0 Analysis Data**: Complete F0 data for all files
- **Summary by Subject**: F0 statistics per subject
- **Summary by Subject & Week**: Longitudinal F0 tracking
- **Summary by Treatment Type**: F0 grouped by treatment
- **Summary by Week**: Weekly F0 statistics
- **Sampling Rate Summary**: Distribution of sampling rates
- **Overall Summary**: High-level F0 statistics

### 3. Visualizations
- `1_audio_count_per_week_by_subject.png`
- `2_total_audio_per_subject.png`
- `3_duration_distribution_by_subject.png`
- `4_total_duration_per_subject.png`
- `5_heatmap_subject_vs_week.png`

## How It Works

1. **Authentication**: Connects to Box using OAuth 2.0
2. **File Discovery**:
   - Finds 8 subject folders ending with "-DONE"
   - Within each, finds folders matching "Processed" patterns
   - Recursively scans for .wav files
3. **Metadata Analysis**: Downloads and analyzes audio file properties
4. **F0 Analysis**: Computes fundamental frequency using autocorrelation
5. **Report Generation**: Creates Excel files and visualizations

## Treatment Types

- **S (Scar)**: Elvis, Cher and Adele
- **B (Bilateral)**: Frank and Dean, Madonna and Beyonce, Beck and Kurt, Taylor and Gaga
- **U (Unilateral)**: Michael and Prince, Paul and John

## Subject Folders

The analysis processes these 8 folders:
1. P24-1-Elvis-DONE
2. P24-2 and 24-3 Frank and Dean-DONE
3. P24-4 & 5 Madonna and Beyonce-DONE
4. P24-6 & 7 Beck and Kurt-DONE
5. P24-8 & 9 Cher and Adele-DONE
6. P24-10 & 11 Michael and Prince-DONE
7. P24-12 & 13 Taylor and Gaga-DONE
8. P24-14 &15 Paul and John-DONE

## Notes

- Only analyzes files within "Processed" folders (or variations)
- No files are modified, only read for analysis
- Automatic token refresh for long-running sessions
- Progress updates displayed during analysis
- Error handling for individual file failures

## Troubleshooting

**Authentication issues:**
- Verify Client ID and Secret are correct
- Ensure redirect URI matches Box app settings
- Check that credentials file is properly formatted

**No files found:**
- Verify folder names match exactly (including "-DONE" suffix)
- Check that "Processed" folders exist in subject folders
- Ensure you have access permissions to the Box folders

**Analysis errors:**
- Check that .wav files are valid audio files
- Verify sufficient memory for large files
- Review console output for specific error messages
