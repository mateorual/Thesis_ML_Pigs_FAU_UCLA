"""
F0 (Fundamental Frequency) Analysis Module
Computes fundamental frequency of audio files using autocorrelation method
"""

import numpy as np
import pandas as pd
from scipy.io import wavfile
from io import BytesIO
import warnings

warnings.filterwarnings('ignore')


def calculate_pitch(audio_in, fs, snippet_duration=0.03, overlap=0.5, f0_min=50, f0_max=500):
    """
    Calculate fundamental frequency using autocorrelation method.

    This is the SAME method used in the original analysis scripts.

    Args:
        audio_in: Audio signal as numpy array
        fs: Sampling frequency (Hz)
        snippet_duration: Duration of each snippet in seconds (default: 0.03s = 30ms)
        overlap: Overlap fraction between snippets (default: 0.5 = 50%)
        f0_min: Minimum valid F0 in Hz (default: 50)
        f0_max: Maximum valid F0 in Hz (default: 500)

    Returns:
        Mean fundamental frequency (Hz), or 0 if no valid F0 detected
    """
    # Remove DC offset
    audio_in = audio_in - np.mean(audio_in)

    snippet_length = int(fs * snippet_duration)
    step_size = int(snippet_length * (1 - overlap))
    f0_values = []

    # Process audio in overlapping snippets
    for start in range(0, len(audio_in) - snippet_length, step_size):
        snippet = audio_in[start:start + snippet_length]

        # Autocorrelation
        corr = np.correlate(snippet, snippet, mode='full')
        corr = corr[len(corr)//2:]  # Keep only positive lags

        # Find first positive derivative (first local minimum)
        d = np.diff(corr)
        pos_indices = np.where(d > 0)[0]

        if len(pos_indices) == 0:
            continue

        start_idx = pos_indices[0]

        # Find peak after the first minimum
        peak_idx = np.argmax(corr[start_idx:]) + start_idx

        # Calculate F0 from peak location
        if peak_idx > 0:
            f0 = fs / peak_idx
            # Filter to reasonable vocal range
            if f0_min <= f0 <= f0_max:
                f0_values.append(f0)

    # Return mean F0 or 0 if no valid values found
    return np.mean(f0_values) if f0_values else 0


def compute_f0_from_bytes(audio_bytes, snippet_duration=0.03, overlap=0.5, f0_min=50, f0_max=500):
    """
    Compute F0 from audio file in BytesIO format

    Args:
        audio_bytes: BytesIO object containing WAV file
        snippet_duration: Snippet duration for analysis
        overlap: Overlap fraction
        f0_min: Minimum valid F0 in Hz
        f0_max: Maximum valid F0 in Hz

    Returns:
        Dictionary with F0 statistics
    """
    try:
        # Read the WAV file using scipy
        fs, audio_in = wavfile.read(audio_bytes)

        # Convert stereo to mono if necessary
        if audio_in.ndim > 1:
            audio_in = audio_in.mean(axis=1)

        # Normalize to -1 to 1
        if np.max(np.abs(audio_in)) > 0:
            audio_in = audio_in / np.max(np.abs(audio_in))
        else:
            # Silent file
            return {
                'F0_mean': None,
                'F0_valid': False,
                'F0_count': 0,
                'Sampling_Rate_Hz': fs
            }

        # Check if file is too short
        if len(audio_in) < fs * 0.1:  # Less than 100ms
            return {
                'F0_mean': None,
                'F0_valid': False,
                'F0_count': 0,
                'Sampling_Rate_Hz': fs
            }

        # Calculate pitch
        f0_mean = calculate_pitch(audio_in, fs, snippet_duration, overlap, f0_min, f0_max)

        if f0_mean == 0:
            return {
                'F0_mean': None,
                'F0_valid': False,
                'F0_count': 0,
                'Sampling_Rate_Hz': fs
            }

        return {
            'F0_mean': round(f0_mean, 2),
            'F0_valid': True,
            'F0_count': 1,
            'Sampling_Rate_Hz': fs
        }

    except Exception as e:
        print(f"Error computing F0: {e}")
        return {
            'F0_mean': None,
            'F0_valid': False,
            'F0_count': 0,
            'Sampling_Rate_Hz': None
        }


def analyze_audio_f0(box_client, file_info, f0_params):
    """
    Analyze F0 for a single audio file

    Args:
        box_client: BoxClientWrapper instance
        file_info: Dictionary with file metadata from Box
        f0_params: Dictionary with F0 analysis parameters

    Returns:
        Dictionary with F0 analysis results
    """
    # Download file to analyze
    audio_bytes = box_client.download_file_to_bytes(file_info['file_id'])

    if audio_bytes is None:
        return None

    # Compute F0
    f0_stats = compute_f0_from_bytes(
        audio_bytes,
        snippet_duration=f0_params['snippet_duration'],
        overlap=f0_params['overlap'],
        f0_min=f0_params['f0_min'],
        f0_max=f0_params['f0_max']
    )

    # Extract week - use provided week if available, otherwise extract from path
    week = file_info.get('week', box_client.extract_week_from_path(file_info['path_info']))

    return {
        'File Name': file_info['file_name'],
        'Week': week,
        'Parent Folder': file_info['parent_folder_name'],
        'Path': file_info['path_info'],
        **f0_stats
    }


def generate_f0_summary(df):
    """
    Generate summary statistics for F0 analysis

    Args:
        df: DataFrame with F0 analysis results

    Returns:
        Dictionary with summary statistics
    """
    summary = {}

    # Overall F0 statistics
    valid_f0 = df[df['F0_valid'] == True]

    summary['Total Files Analyzed'] = len(df)
    summary['Files with Valid F0'] = len(valid_f0)
    summary['Files without Valid F0'] = len(df) - len(valid_f0)
    summary['Valid F0 Percentage'] = round((len(valid_f0) / len(df) * 100), 2) if len(df) > 0 else 0

    if len(valid_f0) > 0:
        summary['Mean F0 (Hz)'] = round(valid_f0['F0_mean'].mean(), 2)
        summary['Median F0 (Hz)'] = round(valid_f0['F0_mean'].median(), 2)
        summary['F0 Standard Deviation (Hz)'] = round(valid_f0['F0_mean'].std(), 2)
        summary['F0 Range (Hz)'] = f"{valid_f0['F0_mean'].min():.2f} - {valid_f0['F0_mean'].max():.2f}"

    return summary


def print_f0_statistics(df):
    """
    Print F0 statistics to console

    Args:
        df: DataFrame with F0 analysis results
    """
    print("\n" + "=" * 80)
    print("F0 ANALYSIS SUMMARY STATISTICS")
    print("=" * 80)

    valid_f0 = df[df['F0_valid'] == True]

    if len(valid_f0) > 0:
        print(f"\nOverall F0 Statistics (Hz):")
        print(f"  Mean F0: {valid_f0['F0_mean'].mean():.2f} Hz")
        print(f"  Median F0: {valid_f0['F0_mean'].median():.2f} Hz")
        print(f"  Std Dev F0: {valid_f0['F0_mean'].std():.2f} Hz")
        print(f"  F0 Range: {valid_f0['F0_mean'].min():.2f} - {valid_f0['F0_mean'].max():.2f} Hz")
        print(f"  Files with valid F0: {len(valid_f0)} / {len(df)} ({len(valid_f0)/len(df)*100:.1f}%)")

    # Treatment type statistics
    if 'Type' in df.columns and len(valid_f0) > 0:
        print(f"\nF0 Statistics by Treatment Type:")
        type_stats = valid_f0.groupby('Type')['F0_mean'].agg(['mean', 'std', 'min', 'max', 'count'])
        type_stats = type_stats.round(2)
        print(type_stats)
        print("\nTreatment Type Key:")
        print("  S = Sham treatment")
        print("  B = Bilateral treatment")
        print("  U = Unilateral treatment")

    # Per-subject statistics
    if len(valid_f0) > 0:
        print(f"\nF0 Statistics by Subject:")
        subject_stats = valid_f0.groupby('Subject')['F0_mean'].agg(['mean', 'std', 'min', 'max', 'count'])
        subject_stats = subject_stats.round(2)
        print(subject_stats)


def save_f0_to_excel(df, output_path):
    """
    Save F0 analysis to Excel file with multiple sheets

    Args:
        df: DataFrame with F0 analysis results
        output_path: Path to save Excel file
    """
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        # Main data sheet
        df_sorted = df.sort_values(['Subject', 'Week', 'File Name'])
        df_sorted.to_excel(writer, sheet_name='F0 Analysis Data', index=False)

        # Filter for valid F0 values
        valid_df = df_sorted[df_sorted['F0_valid'] == True]

        if len(valid_df) > 0:
            # Summary by subject
            subject_summary = valid_df.groupby('Subject').agg({
                'File Name': 'count',
                'F0_mean': ['mean', 'std', 'min', 'max'],
                'Sampling_Rate_Hz': lambda x: x.mode()[0] if len(x.mode()) > 0 else None
            }).round(2)
            subject_summary.columns = ['File Count', 'Mean F0 (Hz)', 'Std F0 (Hz)',
                                       'Min F0 (Hz)', 'Max F0 (Hz)', 'Common Sampling Rate']
            subject_summary.to_excel(writer, sheet_name='Summary by Subject')

            # Summary by subject and week
            if valid_df['Week'].notna().any():
                subject_week_df = valid_df[valid_df['Week'].notna()].copy()
                subject_week_summary = subject_week_df.groupby(['Subject', 'Week']).agg({
                    'File Name': 'count',
                    'F0_mean': ['mean', 'std'],
                    'Type': 'first'
                }).round(2)
                subject_week_summary.columns = ['File Count', 'Mean F0 (Hz)', 'Std Dev F0 (Hz)', 'Type']
                subject_week_summary = subject_week_summary.reset_index()
                subject_week_summary = subject_week_summary[['Subject', 'Type', 'Week', 'File Count', 'Mean F0 (Hz)', 'Std Dev F0 (Hz)']]
                subject_week_summary.to_excel(writer, sheet_name='Summary by Subject & Week', index=False)

            # Summary by treatment type
            if 'Type' in valid_df.columns:
                type_summary = valid_df.groupby('Type').agg({
                    'File Name': 'count',
                    'F0_mean': ['mean', 'std', 'min', 'max'],
                    'Subject': 'nunique'
                }).round(2)
                type_summary.columns = ['File Count', 'Mean F0 (Hz)', 'Std F0 (Hz)',
                                       'Min F0 (Hz)', 'Max F0 (Hz)', 'Number of Subjects']
                type_summary.to_excel(writer, sheet_name='Summary by Treatment Type')

            # Summary by week
            if valid_df['Week'].notna().any():
                week_summary = valid_df[valid_df['Week'].notna()].groupby('Week').agg({
                    'File Name': 'count',
                    'F0_mean': ['mean', 'std', 'min', 'max']
                }).round(2)
                week_summary.columns = ['File Count', 'Mean F0 (Hz)', 'Std F0 (Hz)',
                                       'Min F0 (Hz)', 'Max F0 (Hz)']
                week_summary.to_excel(writer, sheet_name='Summary by Week')

        # Sampling rate summary
        if df['Sampling_Rate_Hz'].notna().any():
            sr_summary = df.groupby('Sampling_Rate_Hz').agg({
                'File Name': 'count',
                'Subject': 'nunique'
            })
            sr_summary.columns = ['File Count', 'Number of Subjects']
            sr_summary.to_excel(writer, sheet_name='Sampling Rate Summary')

        # Overall summary
        summary = generate_f0_summary(df)
        summary_df = pd.DataFrame(list(summary.items()), columns=['Metric', 'Value'])
        summary_df.to_excel(writer, sheet_name='Overall Summary', index=False)

    print(f"F0 analysis saved to: {output_path}")
