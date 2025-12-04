"""
Main script for Box Basic Data Analysis
Orchestrates the entire analysis pipeline:
1. Authenticate with Box
2. Navigate to subject folders
3. Find "Processed" folders
4. Analyze audio files (metadata and F0)
5. Generate reports and visualizations
"""

import os
import pandas as pd
from datetime import datetime

# Import configuration
import config

# Import helpers
from helpers import authenticate_or_refresh, BoxClientWrapper, save_credentials

# Import analysis modules
from analysis import (
    analyze_audio_metadata,
    create_metadata_visualizations,
    save_metadata_to_excel,
    analyze_audio_f0,
    print_f0_statistics,
    save_f0_to_excel
)


def find_and_process_audio_files(box_client, root_folder_id, main_folders, processed_patterns):
    """
    Find all audio files in "Processed" folders within subject folders

    Args:
        box_client: BoxClientWrapper instance
        root_folder_id: Root folder ID containing subject folders
        main_folders: List of subject folder names to process
        processed_patterns: List of "Processed" folder name patterns

    Returns:
        Dictionary mapping subject_name -> list of file_info dictionaries
    """
    print("\n" + "=" * 80)
    print("SCANNING FOR AUDIO FILES")
    print("=" * 80)

    # Find main subject folders
    print(f"\nSearching for subject folders in root folder {root_folder_id}...")
    subject_folders = box_client.find_folders_by_name(root_folder_id, main_folders)

    print(f"Found {len(subject_folders)} subject folders:")
    for folder_name in subject_folders:
        print(f"  - {folder_name}")

    # Process each subject folder
    all_files_by_subject = {}

    for folder_name, folder_id in subject_folders.items():
        subject_name = box_client.extract_subject_name(folder_name)
        print(f"\n{'='*60}")
        print(f"Processing Subject: {subject_name}")
        print(f"{'='*60}")

        # Find "Processed" folders within this subject folder
        processed_folders = box_client.find_processed_folders(folder_id, processed_patterns)

        if not processed_folders:
            print(f"  No 'Processed' folders found for {subject_name}")
            continue

        print(f"  Found {len(processed_folders)} 'Processed' folder(s):")
        for proc_name, proc_id in processed_folders:
            print(f"    - {proc_name} (ID: {proc_id})")

        # Get all WAV files from all processed folders
        subject_files = []
        for proc_name, proc_id in processed_folders:
            print(f"  Scanning '{proc_name}' for audio files...")
            wav_files = box_client.get_all_wav_files_recursive(proc_id, config.AUDIO_EXTENSION)
            subject_files.extend(wav_files)

        print(f"  Total audio files found for {subject_name}: {len(subject_files)}")
        all_files_by_subject[subject_name] = subject_files

    return all_files_by_subject


def find_and_process_snip_files(box_client, all_pig_snips_folder_id, treatment_folders_dict, target_subjects):
    """
    Find all audio files in "All Pig Snips" structure

    Args:
        box_client: BoxClientWrapper instance
        all_pig_snips_folder_id: ID of "All Pig Snips" folder
        treatment_folders_dict: Dictionary mapping treatment code -> folder name
        target_subjects: List of subject names to analyze

    Returns:
        Dictionary mapping subject_name -> list of file_info dictionaries
    """
    print("\n" + "=" * 80)
    print("SCANNING FOR AUDIO FILES (All Pig Snips Structure)")
    print("=" * 80)

    # Get snip folders organized by treatment
    snip_folders_by_subject = box_client.get_snip_folders_by_treatment(
        all_pig_snips_folder_id,
        treatment_folders_dict,
        target_subjects
    )

    # Process each subject's snip folders
    all_files_by_subject = {}

    for subject_name, snip_folders in snip_folders_by_subject.items():
        if not snip_folders:
            print(f"\n{'='*60}")
            print(f"Processing Subject: {subject_name}")
            print(f"{'='*60}")
            print(f"  No snip folders found for {subject_name}")
            continue

        print(f"\n{'='*60}")
        print(f"Processing Subject: {subject_name}")
        print(f"{'='*60}")
        print(f"  Found {len(snip_folders)} snip folder(s)")

        subject_files = []

        for snip_info in snip_folders:
            folder_name = snip_info['folder_name']
            folder_id = snip_info['folder_id']
            week = snip_info['week']

            print(f"  Scanning '{folder_name}' for audio files...")

            # Get all WAV files from this snip folder (recursively)
            wav_files = box_client.get_all_wav_files_with_metadata(
                folder_id,
                subject_name,
                week,
                config.AUDIO_EXTENSION
            )

            subject_files.extend(wav_files)

        print(f"  Total audio files found for {subject_name}: {len(subject_files)}")
        all_files_by_subject[subject_name] = subject_files

    return all_files_by_subject


def analyze_metadata(box_client, files_by_subject, treatment_mapping):
    """
    Analyze metadata for all audio files

    Args:
        box_client: BoxClientWrapper instance
        files_by_subject: Dictionary mapping subject -> list of file_info
        treatment_mapping: Dictionary mapping subject -> treatment type

    Returns:
        DataFrame with metadata analysis results
    """
    print("\n" + "=" * 80)
    print("METADATA ANALYSIS")
    print("=" * 80)

    all_metadata = []
    total_files = sum(len(files) for files in files_by_subject.values())
    processed = 0

    for subject_name, files in files_by_subject.items():
        print(f"\nAnalyzing metadata for {subject_name} ({len(files)} files)...")
        treatment_type = treatment_mapping.get(subject_name, "Unknown")

        for file_info in files:
            try:
                metadata = analyze_audio_metadata(box_client, file_info)
                if metadata:
                    metadata['Subject'] = subject_name
                    metadata['Type'] = treatment_type
                    all_metadata.append(metadata)

                processed += 1
                if processed % 10 == 0:
                    print(f"  Progress: {processed}/{total_files} files analyzed")

            except Exception as e:
                print(f"  Error analyzing {file_info['file_name']}: {e}")

    df_metadata = pd.DataFrame(all_metadata)
    print(f"\nMetadata analysis complete: {len(df_metadata)} files processed successfully")

    return df_metadata


def analyze_f0(box_client, files_by_subject, treatment_mapping, f0_params):
    """
    Analyze F0 for all audio files

    Args:
        box_client: BoxClientWrapper instance
        files_by_subject: Dictionary mapping subject -> list of file_info
        treatment_mapping: Dictionary mapping subject -> treatment type
        f0_params: Dictionary with F0 analysis parameters

    Returns:
        DataFrame with F0 analysis results
    """
    print("\n" + "=" * 80)
    print("F0 ANALYSIS")
    print("=" * 80)
    print(f"Method: Autocorrelation-based pitch detection")
    print(f"F0 Range: {f0_params['f0_min']}-{f0_params['f0_max']} Hz")
    print(f"Snippet Duration: {f0_params['snippet_duration']*1000}ms")
    print(f"Overlap: {f0_params['overlap']*100}%")

    all_f0_results = []
    total_files = sum(len(files) for files in files_by_subject.values())
    processed = 0

    for subject_name, files in files_by_subject.items():
        print(f"\nAnalyzing F0 for {subject_name} ({len(files)} files)...")
        treatment_type = treatment_mapping.get(subject_name, "Unknown")

        for file_info in files:
            try:
                f0_result = analyze_audio_f0(box_client, file_info, f0_params)
                if f0_result:
                    f0_result['Subject'] = subject_name
                    f0_result['Type'] = treatment_type
                    all_f0_results.append(f0_result)

                processed += 1
                if processed % 10 == 0:
                    print(f"  Progress: {processed}/{total_files} files analyzed")

            except Exception as e:
                print(f"  Error analyzing F0 for {file_info['file_name']}: {e}")

    df_f0 = pd.DataFrame(all_f0_results)
    print(f"\nF0 analysis complete: {len(df_f0)} files processed successfully")

    return df_f0


def main():
    """Main execution function"""
    print("=" * 80)
    print("BOX BASIC DATA ANALYSIS")
    print("=" * 80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Step 1: Authenticate with Box
    print("\n" + "=" * 80)
    print("STEP 1: AUTHENTICATION")
    print("=" * 80)

    credentials = authenticate_or_refresh(config.CREDENTIALS_FILE)
    if credentials is None:
        print("\nAuthentication failed. Exiting.")
        return

    # Create Box client
    box_client = BoxClientWrapper(credentials)

    # Verify authentication
    try:
        user = box_client.get_user_info()
        print(f"\nAuthenticated as: {user.name} ({user.login})")
    except Exception as e:
        print(f"\nAuthentication verification failed: {e}")
        return

    # Save potentially refreshed credentials
    save_credentials(credentials, config.CREDENTIALS_FILE)

    # Step 2: Find and scan audio files based on folder structure mode
    print("\n" + "=" * 80)
    print(f"FOLDER STRUCTURE MODE: {config.FOLDER_STRUCTURE}")
    print("=" * 80)

    if config.FOLDER_STRUCTURE == "Individual Pairs DONE":
        # Mode 1: Individual subject folders with "Processed" subfolders
        files_by_subject = find_and_process_audio_files(
            box_client,
            config.BOX_ROOT_FOLDER_ID,
            config.MAIN_FOLDERS,
            config.PROCESSED_FOLDER_PATTERNS
        )
    elif config.FOLDER_STRUCTURE == "All Pig Snips":
        # Mode 2: All Pig Snips structure with treatment folders
        target_subjects = list(config.TREATMENT_MAPPING.keys())
        files_by_subject = find_and_process_snip_files(
            box_client,
            config.ALL_PIG_SNIPS_FOLDER_ID,
            config.ALL_PIG_SNIPS_TREATMENT_FOLDERS,
            target_subjects
        )
    else:
        print(f"\nError: Invalid FOLDER_STRUCTURE setting: '{config.FOLDER_STRUCTURE}'")
        print("Valid options: 'Individual Pairs DONE' or 'All Pig Snips'")
        return

    total_files = sum(len(files) for files in files_by_subject.values())
    if total_files == 0:
        print("\nNo audio files found. Exiting.")
        return

    print(f"\n{'='*80}")
    print(f"TOTAL AUDIO FILES FOUND: {total_files}")
    print(f"{'='*80}")

    # Display analysis type
    print("\n" + "=" * 80)
    print(f"ANALYSIS TYPE: {config.ANALYSIS_TYPE}")
    print("=" * 80)

    # Validate analysis type
    valid_analysis_types = ["Metadata Analysis", "F0 Analysis", "Metadata and F0 Analysis"]
    if config.ANALYSIS_TYPE not in valid_analysis_types:
        print(f"\nError: Invalid ANALYSIS_TYPE setting: '{config.ANALYSIS_TYPE}'")
        print(f"Valid options: {', '.join(valid_analysis_types)}")
        return

    # Step 3: Run selected analyses
    df_metadata = None
    df_f0 = None

    # Metadata Analysis
    if config.ANALYSIS_TYPE in ["Metadata Analysis", "Metadata and F0 Analysis"]:
        df_metadata = analyze_metadata(box_client, files_by_subject, config.TREATMENT_MAPPING)

    # F0 Analysis
    if config.ANALYSIS_TYPE in ["F0 Analysis", "Metadata and F0 Analysis"]:
        f0_params = {
            'snippet_duration': config.F0_SNIPPET_DURATION,
            'overlap': config.F0_OVERLAP,
            'f0_min': config.F0_MIN_FREQ,
            'f0_max': config.F0_MAX_FREQ
        }
        df_f0 = analyze_f0(box_client, files_by_subject, config.TREATMENT_MAPPING, f0_params)

    # Step 4: Generate outputs
    print("\n" + "=" * 80)
    print("GENERATING OUTPUTS")
    print("=" * 80)

    # Create output directory
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    viz_files = []

    # Save metadata analysis if performed
    if df_metadata is not None:
        metadata_excel_path = os.path.join(config.OUTPUT_DIR, config.METADATA_EXCEL_FILENAME)
        save_metadata_to_excel(df_metadata, metadata_excel_path)

        # Create visualizations
        print("\nCreating visualizations...")
        viz_files = create_metadata_visualizations(df_metadata, config.OUTPUT_DIR)
        print(f"Created {len(viz_files)} visualizations")

    # Save F0 analysis if performed
    if df_f0 is not None:
        f0_excel_path = os.path.join(config.OUTPUT_DIR, config.F0_EXCEL_FILENAME)
        save_f0_to_excel(df_f0, f0_excel_path)

        # Print F0 statistics
        print_f0_statistics(df_f0)

    # Final summary
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE!")
    print("=" * 80)
    print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nOutput files saved to: {config.OUTPUT_DIR}")

    # Display generated files based on what was run
    if df_metadata is not None:
        print(f"  - Metadata Excel: {config.METADATA_EXCEL_FILENAME}")
        print(f"  - Visualizations: {len(viz_files)} PNG files")

    if df_f0 is not None:
        print(f"  - F0 Excel: {config.F0_EXCEL_FILENAME}")

    # List visualization files if created
    if viz_files:
        print("\nVisualization files created:")
        for viz in viz_files:
            print(f"  - {os.path.basename(viz)}")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
