"""
Audio Metadata Analysis Module
Extracts and analyzes basic metadata from audio files (duration, size, etc.)
"""

import wave
import struct
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import timedelta
from io import BytesIO
import sys
import os

# Add parent directory to path to import config
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# Set style for plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


def parse_wav_header(header_bytes):
    """
    Parse WAV file header to extract metadata (FAST - no full file download)

    Args:
        header_bytes: BytesIO object containing first ~500 bytes of WAV file

    Returns:
        Dictionary with duration and sampling_rate, or None if parsing fails
    """
    try:
        header_bytes.seek(0)
        data = header_bytes.read(512)  # Read up to 512 bytes

        # Check RIFF header
        if data[0:4] != b'RIFF' or data[8:12] != b'WAVE':
            return None

        # Find fmt chunk
        pos = 12
        while pos < len(data) - 8:
            chunk_id = data[pos:pos+4]
            chunk_size = struct.unpack('<I', data[pos+4:pos+8])[0]

            if chunk_id == b'fmt ':
                # Parse fmt chunk
                fmt_data = data[pos+8:pos+8+chunk_size]
                if len(fmt_data) >= 16:
                    audio_format = struct.unpack('<H', fmt_data[0:2])[0]
                    num_channels = struct.unpack('<H', fmt_data[2:4])[0]
                    sample_rate = struct.unpack('<I', fmt_data[4:8])[0]
                    byte_rate = struct.unpack('<I', fmt_data[8:12])[0]
                    block_align = struct.unpack('<H', fmt_data[12:14])[0]
                    bits_per_sample = struct.unpack('<H', fmt_data[14:16])[0]

                    # Look for data chunk to get size
                    data_pos = pos + 8 + chunk_size
                    while data_pos < len(data) - 8:
                        data_chunk_id = data[data_pos:data_pos+4]
                        if data_chunk_id == b'data':
                            data_size = struct.unpack('<I', data[data_pos+4:data_pos+8])[0]

                            # Calculate duration
                            if byte_rate > 0:
                                duration = data_size / byte_rate
                                return {
                                    'duration': duration,
                                    'sampling_rate': sample_rate,
                                    'channels': num_channels,
                                    'bits_per_sample': bits_per_sample
                                }
                            break
                        data_pos += 8 + struct.unpack('<I', data[data_pos+4:data_pos+8])[0]
                    break

            pos += 8 + chunk_size

        return None

    except Exception as e:
        # If header parsing fails, return None (will fallback to full file download)
        return None


def get_audio_duration_from_bytes(audio_bytes):
    """
    Get duration of WAV file from BytesIO object

    Args:
        audio_bytes: BytesIO object containing WAV file

    Returns:
        Duration in seconds or None if error
    """
    try:
        with wave.open(audio_bytes, 'rb') as wav_file:
            frames = wav_file.getnframes()
            rate = wav_file.getframerate()
            duration = frames / float(rate)
            return duration
    except Exception as e:
        print(f"Error reading audio duration: {e}")
        return None


def get_sampling_rate_from_bytes(audio_bytes):
    """
    Get sampling rate of WAV file from BytesIO object

    Args:
        audio_bytes: BytesIO object containing WAV file

    Returns:
        Sampling rate in Hz or None if error
    """
    try:
        with wave.open(audio_bytes, 'rb') as wav_file:
            return wav_file.getframerate()
    except Exception as e:
        print(f"Error reading sampling rate: {e}")
        return None


def analyze_audio_metadata(box_client, file_info):
    """
    Analyze metadata for a single audio file (OPTIMIZED - tries header-only first)

    Args:
        box_client: BoxClientWrapper instance
        file_info: Dictionary with file metadata from Box

    Returns:
        Dictionary with extended metadata including duration
    """
    duration = None
    sampling_rate = None

    # OPTIMIZATION: Try header-only download first (much faster!)
    header_bytes = box_client.download_file_header(file_info['file_id'])

    if header_bytes is not None:
        # Try to parse header
        header_info = parse_wav_header(header_bytes)
        if header_info is not None:
            duration = header_info['duration']
            sampling_rate = header_info['sampling_rate']

    # FALLBACK: If header parsing failed, download full file
    if duration is None or sampling_rate is None:
        audio_bytes = box_client.download_file_to_bytes(file_info['file_id'])

        if audio_bytes is None:
            return None

        # Get duration using wave module
        if duration is None:
            duration = get_audio_duration_from_bytes(audio_bytes)

        # Get sampling rate
        if sampling_rate is None:
            audio_bytes.seek(0)
            sampling_rate = get_sampling_rate_from_bytes(audio_bytes)

    # Convert size to MB
    size_mb = file_info['size_bytes'] / (1024 * 1024)

    # Extract week - use provided week if available, otherwise extract from path
    week = file_info.get('week', box_client.extract_week_from_path(file_info['path_info']))

    return {
        'File Name': file_info['file_name'],
        'Duration (seconds)': round(duration, 2) if duration else None,
        'Duration (mm:ss)': str(timedelta(seconds=int(duration))) if duration else None,
        'Size (MB)': round(size_mb, 2),
        'Sampling Rate (Hz)': sampling_rate,
        'Week': week,
        'Parent Folder': file_info['parent_folder_name'],
        'Path': file_info['path_info']
    }


def create_color_mapping(subjects):
    """Create consistent color mapping for all subjects"""
    colors = sns.color_palette("muted", len(subjects))
    sorted_subjects = sorted(subjects)
    color_map = {subject: colors[i] for i, subject in enumerate(sorted_subjects)}
    return color_map


def create_metadata_visualizations(df, output_dir, subject_col='Subject'):
    """
    Create visualizations for metadata analysis

    Args:
        df: DataFrame with metadata
        output_dir: Directory to save visualizations
        subject_col: Column name for subject

    Returns:
        List of created visualization file paths
    """
    import os

    visualization_files = []

    # Create consistent color mapping
    all_subjects = sorted(df[subject_col].unique())
    color_map = create_color_mapping(all_subjects)

    # 1. Audio count per week by subject
    df_with_weeks = df[df['Week'].notna()].copy()

    if not df_with_weeks.empty:
        plt.figure(figsize=(14, 8))
        subjects = sorted(df_with_weeks[subject_col].unique())
        width = 0.1
        x = np.arange(len(df_with_weeks['Week'].unique()))

        fig, ax = plt.subplots(figsize=(14, 8))

        for i, subject in enumerate(subjects):
            subject_data = df_with_weeks[df_with_weeks[subject_col] == subject]
            week_counts = subject_data.groupby('Week').size()

            all_weeks = sorted(df_with_weeks['Week'].unique())
            counts = [week_counts.get(week, 0) for week in all_weeks]

            ax.bar(x + i * width, counts, width, label=subject, color=color_map[subject])

        ax.set_xlabel('Week', fontsize=12, fontweight='bold')
        ax.set_ylabel('Number of Audio Files', fontsize=12, fontweight='bold')
        ax.set_title('Number of Audio Files per Week by Subject', fontsize=14, fontweight='bold')
        ax.set_xticks(x + width * (len(subjects) - 1) / 2)
        ax.set_xticklabels([f'Week {w}' for w in all_weeks])
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        filename = os.path.join(output_dir, '1_audio_count_per_week_by_subject.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        visualization_files.append(filename)
        plt.close()

    # 2. Total audios per subject (Stacked: Pre vs Post)
    plt.figure(figsize=(10, 6))

    # Separate pre-op (week 0) and post-op (weeks 1-27)
    df_with_weeks_plot2 = df[df['Week'].notna()].copy()

    pre_counts = {}
    post_counts = {}
    all_subjects_plot2 = sorted(df_with_weeks_plot2[subject_col].unique())

    for subject in all_subjects_plot2:
        subject_data = df_with_weeks_plot2[df_with_weeks_plot2[subject_col] == subject]
        pre_counts[subject] = len(subject_data[subject_data['Week'] == 0])
        post_counts[subject] = len(subject_data[subject_data['Week'] > 0])

    # Sort by total count
    total_counts = {s: pre_counts[s] + post_counts[s] for s in all_subjects_plot2}
    sorted_subjects = sorted(all_subjects_plot2, key=lambda x: total_counts[x], reverse=True)

    pre_values = [pre_counts[s] for s in sorted_subjects]
    post_values = [post_counts[s] for s in sorted_subjects]
    colors = [color_map[subject] for subject in sorted_subjects]

    x = range(len(sorted_subjects))

    # Create stacked bars with different patterns
    bars1 = plt.bar(x, pre_values, color=colors, edgecolor='black', linewidth=1.5, label='Pre-op (Week 0)')
    bars2 = plt.bar(x, post_values, bottom=pre_values, color=colors, edgecolor='black', linewidth=1.5,
                    hatch='///', label='Post-op (Weeks 1-27)')

    plt.xlabel('Subject', fontsize=12, fontweight='bold')
    plt.ylabel('Total Number of Audio Files', fontsize=12, fontweight='bold')
    plt.title('Total Audio Files per Subject (Pre vs Post Operation)', fontsize=14, fontweight='bold')
    plt.xticks(x, sorted_subjects, rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    plt.legend(loc='upper right')

    # Add total count labels
    for i, (pre, post) in enumerate(zip(pre_values, post_values)):
        total = pre + post
        plt.text(i, total + 0.5, str(total), ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    filename = os.path.join(output_dir, '2_total_audio_per_subject.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    visualization_files.append(filename)
    plt.close()

    # 3. Duration distribution by subject
    df_with_duration = df[df['Duration (seconds)'].notna()].copy()

    if not df_with_duration.empty:
        plt.figure(figsize=(12, 6))
        median_durations = df_with_duration.groupby(subject_col)['Duration (seconds)'].median().sort_values(ascending=False)
        subjects_sorted = median_durations.index.tolist()

        duration_data = [df_with_duration[df_with_duration[subject_col] == subj]['Duration (seconds)'].values
                         for subj in subjects_sorted]

        bp = plt.boxplot(duration_data, labels=subjects_sorted, patch_artist=True)

        for patch, subject in zip(bp['boxes'], subjects_sorted):
            patch.set_facecolor(color_map[subject])

        plt.xlabel('Subject', fontsize=12, fontweight='bold')
        plt.ylabel('Duration (seconds)', fontsize=12, fontweight='bold')
        plt.title('Distribution of Audio Duration by Subject', fontsize=14, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        filename = os.path.join(output_dir, '3_duration_distribution_by_subject.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        visualization_files.append(filename)
        plt.close()

    # 4. Total duration per subject (Stacked: Pre vs Post)
    plt.figure(figsize=(10, 6))

    # Separate pre-op (week 0) and post-op (weeks 1-27)
    df_with_weeks_plot4 = df[df['Week'].notna() & df['Duration (seconds)'].notna()].copy()

    pre_duration = {}
    post_duration = {}
    all_subjects_plot4 = sorted(df_with_weeks_plot4[subject_col].unique())

    for subject in all_subjects_plot4:
        subject_data = df_with_weeks_plot4[df_with_weeks_plot4[subject_col] == subject]
        pre_duration[subject] = subject_data[subject_data['Week'] == 0]['Duration (seconds)'].sum() / 3600
        post_duration[subject] = subject_data[subject_data['Week'] > 0]['Duration (seconds)'].sum() / 3600

    # Sort by total duration
    total_duration = {s: pre_duration[s] + post_duration[s] for s in all_subjects_plot4}
    sorted_subjects = sorted(all_subjects_plot4, key=lambda x: total_duration[x], reverse=True)

    pre_values = [pre_duration[s] for s in sorted_subjects]
    post_values = [post_duration[s] for s in sorted_subjects]
    colors = [color_map[subject] for subject in sorted_subjects]

    x = range(len(sorted_subjects))

    # Create stacked bars with different patterns
    bars1 = plt.bar(x, pre_values, color=colors, edgecolor='black', linewidth=1.5, label='Pre-op (Week 0)')
    bars2 = plt.bar(x, post_values, bottom=pre_values, color=colors, edgecolor='black', linewidth=1.5,
                    hatch='///', label='Post-op (Weeks 1-27)')

    plt.xlabel('Subject', fontsize=12, fontweight='bold')
    plt.ylabel('Total Duration (hours)', fontsize=12, fontweight='bold')
    plt.title('Total Recording Duration per Subject (Pre vs Post Operation)', fontsize=14, fontweight='bold')
    plt.xticks(x, sorted_subjects, rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    plt.legend(loc='upper right')

    # Add total duration labels
    for i, (pre, post) in enumerate(zip(pre_values, post_values)):
        total = pre + post
        plt.text(i, total + 0.1, f'{total:.2f}h', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    filename = os.path.join(output_dir, '4_total_duration_per_subject.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    visualization_files.append(filename)
    plt.close()

    # 5. Enhanced Heatmap with X for no recordings, 0 for no files
    if not df_with_weeks.empty:
        # Get all unique subjects
        all_subjects = sorted(df_with_weeks[subject_col].unique())
        all_weeks = list(range(28))  # Weeks 0 to 27

        # Create processed weeks mapping from dataframe
        processed_weeks_mapping = {}
        for subject in all_subjects:
            subject_data = df_with_weeks[df_with_weeks[subject_col] == subject]
            processed_weeks_mapping[subject] = subject_data.groupby('Week').size().to_dict()

        # Create matrices for heatmap
        heatmap_data = []
        annotations = []

        for subject in all_subjects:
            raw_weeks = config.RAW_WEEKS_MAPPING.get(subject, [])
            processed_counts = processed_weeks_mapping.get(subject, {})

            subject_row_data = []
            subject_row_annot = []

            for week in all_weeks:
                if week not in raw_weeks:
                    # No recordings taken - mark with X
                    subject_row_data.append(np.nan)  # Use NaN for X
                    subject_row_annot.append('X')
                elif week in processed_counts:
                    # Recordings taken and files exist - show count
                    count = processed_counts[week]
                    subject_row_data.append(count)
                    subject_row_annot.append(str(int(count)))
                else:
                    # Recordings taken but no files - show 0
                    subject_row_data.append(0)
                    subject_row_annot.append('0')

            heatmap_data.append(subject_row_data)
            annotations.append(subject_row_annot)

        # Convert to numpy arrays
        heatmap_data = np.array(heatmap_data)
        annotations = np.array(annotations)

        # Create the heatmap
        fig, ax = plt.subplots(figsize=(18, 8))

        # Create a masked array for NaN values (X marks)
        masked_data = np.ma.masked_invalid(heatmap_data)

        # Plot heatmap
        im = ax.imshow(masked_data, cmap='YlOrRd', aspect='auto', interpolation='nearest')

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Number of Files', fontsize=12, fontweight='bold')

        # Set ticks and labels
        ax.set_xticks(np.arange(len(all_weeks)))
        ax.set_yticks(np.arange(len(all_subjects)))
        ax.set_xticklabels(all_weeks)
        ax.set_yticklabels(all_subjects)

        # Add text annotations and gray background for X
        for i in range(len(all_subjects)):
            for j in range(len(all_weeks)):
                text = annotations[i, j]
                if text == 'X':
                    # Gray background for X
                    ax.add_patch(plt.Rectangle((j-0.5, i-0.5), 1, 1,
                                              fill=True, color='lightgray',
                                              edgecolor='white', linewidth=1.5))
                    ax.text(j, i, text, ha='center', va='center',
                           fontsize=8, fontweight='bold', color='black')
                else:
                    ax.text(j, i, text, ha='center', va='center',
                           fontsize=8, fontweight='bold',
                           color='black')

        # Add grid
        ax.set_xticks(np.arange(len(all_weeks))-0.5, minor=True)
        ax.set_yticks(np.arange(len(all_subjects))-0.5, minor=True)
        ax.grid(which='minor', color='white', linestyle='-', linewidth=1.5)
        ax.tick_params(which='minor', size=0)

        plt.xlabel('Week', fontsize=12, fontweight='bold')
        plt.ylabel('Subject', fontsize=12, fontweight='bold')
        plt.title('Audio File Count Heatmap (Subject vs Week)\nX = No recordings taken | 0 = Recordings had no squeals',
                 fontsize=14, fontweight='bold', pad=20)

        plt.tight_layout()
        filename = os.path.join(output_dir, '5_heatmap_subject_vs_week.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        visualization_files.append(filename)
        plt.close()

    return visualization_files


def generate_metadata_summary(df):
    """
    Generate summary statistics for metadata

    Args:
        df: DataFrame with metadata

    Returns:
        Dictionary with summary statistics
    """
    summary = {}

    summary['Total Audio Files'] = len(df)
    summary['Total Subjects'] = df['Subject'].nunique()
    summary['Total Weeks Covered'] = df['Week'].nunique() if df['Week'].notna().any() else 0

    # Duration statistics
    if df['Duration (seconds)'].notna().any():
        total_duration_hours = df['Duration (seconds)'].sum() / 3600
        avg_duration_seconds = df['Duration (seconds)'].mean()
        summary['Total Duration (hours)'] = round(total_duration_hours, 2)
        summary['Average Duration (seconds)'] = round(avg_duration_seconds, 2)

    # Size statistics
    if df['Size (MB)'].notna().any():
        total_size_gb = df['Size (MB)'].sum() / 1024
        avg_size_mb = df['Size (MB)'].mean()
        summary['Total Data Size (GB)'] = round(total_size_gb, 2)
        summary['Average File Size (MB)'] = round(avg_size_mb, 2)

    return summary


def save_metadata_to_excel(df, output_path):
    """
    Save metadata analysis to Excel file with multiple sheets

    Args:
        df: DataFrame with metadata
        output_path: Path to save Excel file
    """
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        # Main data sheet
        df_sorted = df.sort_values(['Subject', 'Week', 'File Name'])
        df_sorted.to_excel(writer, sheet_name='Audio Files Data', index=False)

        # Summary by subject
        subject_summary = df.groupby('Subject').agg({
            'File Name': 'count',
            'Duration (seconds)': ['sum', 'mean'],
            'Size (MB)': ['sum', 'mean'],
            'Week': 'nunique'
        }).round(2)
        subject_summary.columns = ['File Count', 'Total Duration (s)', 'Avg Duration (s)',
                                   'Total Size (MB)', 'Avg Size (MB)', 'Number of Weeks']
        subject_summary.to_excel(writer, sheet_name='Summary by Subject')

        # Summary by week
        if df['Week'].notna().any():
            week_summary = df[df['Week'].notna()].groupby('Week').agg({
                'File Name': 'count',
                'Duration (seconds)': ['sum', 'mean'],
                'Size (MB)': ['sum', 'mean']
            }).round(2)
            week_summary.columns = ['File Count', 'Total Duration (s)', 'Avg Duration (s)',
                                   'Total Size (MB)', 'Avg Size (MB)']
            week_summary.to_excel(writer, sheet_name='Summary by Week')

        # Overall summary
        summary = generate_metadata_summary(df)
        summary_df = pd.DataFrame(list(summary.items()), columns=['Metric', 'Value'])
        summary_df.to_excel(writer, sheet_name='Overall Summary', index=False)

    print(f"Metadata analysis saved to: {output_path}")
