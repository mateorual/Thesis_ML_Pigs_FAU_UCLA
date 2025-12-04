"""
Configuration file for Box Basic Data Analysis
Contains all constants and configuration parameters
"""

# Box Configuration
BOX_ROOT_FOLDER_ID = '237821336437'
CREDENTIALS_FILE = 'box_credentials.txt'

# Folder Structure Selection
# Options: "Individual Pairs DONE" or "All Pig Snips"
FOLDER_STRUCTURE = "All Pig Snips"  # Change this to switch between analysis modes

# Analysis Type Selection
# Options: "Metadata Analysis", "F0 Analysis", or "Metadata and F0 Analysis"
ANALYSIS_TYPE = "Metadata Analysis"  # Choose which analyses to run

# All Pig Snips Configuration
ALL_PIG_SNIPS_FOLDER_ID = '348140860679'
ALL_PIG_SNIPS_TREATMENT_FOLDERS = {
    'B': 'Bilateral - DONE',    # Bilateral treatment
    'U': 'Unilateral',           # Unilateral treatment
    'S': 'Scar'                  # Scar treatment
}

# Subject folders to analyze (must end with "-DONE")
# Used when FOLDER_STRUCTURE = "Individual Pairs DONE"
MAIN_FOLDERS = [
    "P24-1-Elvis-DONE",
    "P24-2 and  24-3 Frank and Dean-DONE",
    "P24-4 & 5 Madonna and Beyonce-DONE",
    "P24-6 & 7 Beck and Kurt-DONE",
    "P24-8 & 9 Cher and Adele-DONE",
    "P24-10 & 11 Michael and Prince-DONE",
    "P24-12 & 13 Taylor and Gaga-DONE",
    "P24-14 &15 Paul and John-DONE"
]

# Processed folder name variations to search for
PROCESSED_FOLDER_PATTERNS = [
    "Processed",
    "Processed for Analysis",
    "Processed-done",
    "processed",
    "processed for analysis",
    "processed-done"
]

# Audio file extension
AUDIO_EXTENSION = '.wav'

# Treatment type mapping for each subject
TREATMENT_MAPPING = {
    "Elvis": "S",
    "Cher and Adele": "S",
    "Frank and Dean": "B",
    "Madonna and Beyonce": "B",
    "Beck and Kurt": "B",
    "Taylor and Gaga": "B",
    "Michael and Prince": "U",
    "Paul and John": "U"
}

# Week folders inside RAW mapping for each subject
RAW_WEEKS_MAPPING = {
    "Elvis": [0,3,4,8,25,26,27],
    "Cher and Adele": [0,2,12,13,22,23,24],
    "Frank and Dean": [0,15,17,23,24,25,26,27],
    "Madonna and Beyonce": [0,1,2,3,4,5,6,19,20,21,22,26],
    "Beck and Kurt": [0,3,15,17,24,26],
    "Taylor and Gaga": [0,1,12,13,15,20,21,22,27],
    "Michael and Prince": [0,1,8,12,13,14,15,26,27],
    "Paul and John": [0,1,4,5,6,12,15,22,24,26,27]
}


# F0 Analysis Parameters
F0_SNIPPET_DURATION = 0.03  # 30ms
F0_OVERLAP = 0.5  # 50% overlap
F0_MIN_FREQ = 50  # Hz - minimum valid F0
F0_MAX_FREQ = 500  # Hz - maximum valid F0

# Output Configuration
OUTPUT_DIR = 'output'
METADATA_EXCEL_FILENAME = 'box_audio_metadata_analysis.xlsx'
F0_EXCEL_FILENAME = 'box_audio_f0_analysis.xlsx'
