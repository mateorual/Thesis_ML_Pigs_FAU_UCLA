"""
Pig audio feature extraction using directory scanning approach

This version scans directories directly instead of relying on Excel metadata.

Approach:
1. Scan all folders in Audio_Snips/{Treatment}/week_X_{Subject}_{Type}_Snip/
2. Process ALL .wav files found in each folder
3. For each .wav file, find corresponding cycles folder (any folder containing the squeal name)
4. Use the most recent .cycles file in that folder

This is more robust than pattern matching and doesn't depend on Excel metadata accuracy.

Pig audio specifications:
- fs = 44.1 kHz
- window length N = 4096
- frequency resolution = 10.766 Hz
- F0 range: 50-2000 Hz
"""

import sys
import os
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import re
from datetime import datetime
import pickle
from scipy.signal import find_peaks, stft, get_window

# Add local paths
script_dir = Path(__file__).parent
sys.path.append(str(script_dir))
sys.path.append(str(script_dir / "svfel"))

from svfel.core import VowelLoader
from svfel.utils import util
import numpy as np

# Import feature extraction modules
import a_gat
import b_dfg
import c_temporal
import d_spectral
import e_cepstral
import f_lpc

# ============================================================================
# CONFIGURATION FLAG - Set to True to enable MATLAB features (g_avca, h_imf)
# Set to False for faster extraction with 11 features only
# ============================================================================
EXTRACT_MATLAB_FEATURES = False  # <-- Change to True to enable MATLAB features
# ============================================================================

# Optional MATLAB features (only load if enabled)
if EXTRACT_MATLAB_FEATURES:
    try:
        import g_avca
        MATLAB_AVAILABLE = True
        print("✓ g_avca (MATLAB) loaded successfully")
    except Exception as e:
        print(f"Warning: g_avca not available (MATLAB required): {e}")
        MATLAB_AVAILABLE = False

    try:
        import h_imf
        IMF_AVAILABLE = True
        print("✓ h_imf (MATLAB) loaded successfully")
    except Exception as e:
        print(f"Warning: h_imf not available (MATLAB required): {e}")
        IMF_AVAILABLE = False
else:
    MATLAB_AVAILABLE = False
    IMF_AVAILABLE = False
    print("ℹ MATLAB features disabled (EXTRACT_MATLAB_FEATURES = False)")

try:
    import i_ratio
    RATIO_AVAILABLE = True
except Exception as e:
    print(f"Warning: i_ratio not available: {e}")
    RATIO_AVAILABLE = False


# Paths configuration
BASE_PATH = script_dir.parent
AUDIO_SNIPS_FOLDER = BASE_PATH / "Audio_Snips"
CYCLES_FOLDER = BASE_PATH / "Detected_Cycles"  # Now mirrors Audio_Snips structure
OUTPUT_FOLDER = BASE_PATH / "Extracted_Features_Structured"


def preprocess(wav, fs):
    """
    Preprocess audio: highpass filter and normalize

    Parameters:
    -----------
    wav : np.array
        Audio waveform
    fs : int
        Sampling frequency

    Returns:
    --------
    wav : np.array
        Preprocessed waveform
    """
    # Highpass filter at 50Hz (F0 minimum)
    wav = np.hstack([np.zeros(1000), wav, np.zeros(1000)])
    wav = util.fir_highpass(wav.reshape(1, -1), fs, cutoff=50, window='blackmanharris')[0]
    wav = wav[1000:-1000]

    # Normalize wav
    wav = util.mean_center(wav, -1)
    wav = util.maxabs_scale(wav, -1)

    return wav


# ============================================================================
# UCLA PARAMETER CALCULATIONS (F0, Jitter, Shimmer, Q50, Flux)
# Reference: unified_audio_metrics_analysis_reference.py
# ============================================================================

def _calc_pitch(audio_in, fs, snippet_duration=0.03, overlap=0.5):
    audio_in = audio_in - np.mean(audio_in)
    snippet_length = int(fs * snippet_duration)
    step_size = int(snippet_length * (1 - overlap))
    f0_values = []
    for start in range(0, len(audio_in) - snippet_length, step_size):
        snippet = audio_in[start:start + snippet_length]
        corr = np.correlate(snippet, snippet, mode='full')
        corr = corr[len(corr)//2:]
        d = np.diff(corr)
        pos_indices = np.where(d > 0)[0]
        if len(pos_indices) == 0:
            continue
        start_idx = pos_indices[0]
        peak_idx = np.argmax(corr[start_idx:]) + start_idx
        if peak_idx > 0:
            f0 = fs / peak_idx
            if 50 <= f0 <= 2000:
                f0_values.append(f0)
    return float(np.mean(f0_values)) if f0_values else 0.0


def _calc_jitter(audio_in, fs):
    peaks, _ = find_peaks(audio_in, distance=int(0.01 * fs))
    if len(peaks) > 1:
        T0 = np.diff(peaks) / fs
        mean_T0 = np.mean(T0)
        return float(np.mean(np.abs(np.diff(T0))) / mean_T0) if mean_T0 > 0 else 0.0
    return 0.0


def _calc_shimmer(audio_in, fs):
    peaks, _ = find_peaks(audio_in, distance=int(0.01 * fs))
    if len(peaks) > 1:
        peak_amplitudes = audio_in[peaks]
        mean_amp = np.mean(peak_amplitudes)
        return float(np.mean(np.abs(np.diff(peak_amplitudes))) / mean_amp) if mean_amp > 0 else 0.0
    return 0.0


def _calc_q50(audio_in, fs):
    window = get_window('hann', len(audio_in))
    windowed_audio = audio_in * window
    fft_vals = np.fft.rfft(windowed_audio)
    power = np.abs(fft_vals) ** 2
    freqs = np.fft.rfftfreq(len(audio_in), d=1 / fs)
    cum_energy = np.cumsum(power)
    total_energy = cum_energy[-1]
    if total_energy == 0:
        return 0.0
    norm_energy = cum_energy / total_energy
    idx_q50 = np.searchsorted(norm_energy, 0.5)
    return float(freqs[idx_q50]) if idx_q50 < len(freqs) else 0.0


def _calc_flux(audio_in, fs, frame_size=0.025, hop_size=0.010):
    nperseg = int(frame_size * fs)
    noverlap = int((frame_size - hop_size) * fs)
    _, _, Zxx = stft(audio_in, fs=fs, window='hann', nperseg=nperseg,
                     noverlap=noverlap, padded=False, boundary=None)
    mag = np.abs(Zxx)
    if mag.shape[1] < 2:
        return 0.0
    flux = np.sum(np.abs(np.diff(mag, axis=1)), axis=0)
    return float(np.mean(flux))


def extract_ucla_parameters(wav_file_path, savepath):
    """
    Compute F0, Jitter, Shimmer, Q50, and Flux using raw (unpreprocessed) audio,
    matching the computation in unified_audio_metrics_analysis_reference.py.

    Parameters:
    -----------
    wav_file_path : Path or str
        Path to the original .wav file (raw audio, not preprocessed)
    savepath : str
        Output .pickle file path
    """
    from scipy.io import wavfile

    fs, audio_in = wavfile.read(str(wav_file_path))

    if audio_in.size == 0:
        raise ValueError("Empty audio file")

    # Convert stereo to mono (same as reference script)
    if audio_in.ndim > 1:
        audio_in = np.mean(audio_in, axis=1).astype(np.float64)
    else:
        audio_in = audio_in.astype(np.float64)

    metrics = {
        'ucla_f0':      _calc_pitch(audio_in, fs),
        'ucla_jitter':  _calc_jitter(audio_in, fs),
        'ucla_shimmer': _calc_shimmer(audio_in, fs),
        'ucla_q50':     _calc_q50(audio_in, fs),
        'ucla_flux':    _calc_flux(audio_in, fs),
    }
    df = pd.DataFrame([metrics])
    with open(savepath, 'wb') as f:
        pickle.dump(df, f)


# ============================================================================
# SCHLEGEL21 PARAMETER CALCULATIONS
# Source: FileS1.m (Linhart 2015, Tallet 2013, Garcia 2015) and FileS2 (Praat/Linhart HNR)
# All computations use raw (unpreprocessed), amplitude-normalised audio to match the
# original MATLAB/Praat scripts.
#
# Column-name → computation mapping (from FileS1.m):
#   Schlegel21_Q50_2   → ParaValsT(4) = Tallet q50win9(1)  = Q50 at call start (window  2/11)
#   Schlegel21_Q50_10  → ParaValsT(5) = Tallet q50win9(end) = Q50 at call end   (window 10/11)
#   Schlegel21_Q50_min → ParaValsT(6) = min(q50win9)
#   Schlegel21_PF      → ParaValsT(1) = peak frequency from whole-signal energy spectrum
#   Schlegel21_Q25     → ParaValsG(1) = Garcia per-frame Q25, averaged
#   Schlegel21_Dur     → length(y)/Fs
#   Schlegel21_HNR     → Praat cross-correlation HNR (FileS2), minimum_pitch=50 Hz for pigs
# ============================================================================

def _windowed_fft(y, fs, wl, ovl, win_type='hamming'):
    """
    Faithful Python translation of MATLAB windowedFFT(y, Fs, wl, ovl, winType).
    Returns: fft_avg, f, ffts (amplitude spectra), spec (energy spectra)
    """
    hwl = wl // 2 + 1
    shift = max(int(wl * (1 - ovl)), 1)
    n_win = max((len(y) - wl) // shift + 1, 1)

    window = (np.hamming(wl) if win_type == 'hamming' else
              np.hanning(wl) if win_type == 'hanning' else
              np.ones(wl))

    ffts = np.zeros((n_win, hwl))
    spec = np.zeros((n_win, hwl))
    fft_avg = np.zeros(hwl)

    for h in range(n_win):
        seg = y[h * shift: h * shift + wl]
        if len(seg) < wl:
            seg = np.pad(seg, (0, wl - len(seg)))
        winfft0 = np.abs(np.fft.fft(seg.astype(np.float64) * window))
        # amplitude spectrum (one-sided, normalised); double non-DC bins
        ffts[h, :] = winfft0[:hwl] / wl
        ffts[h, 1:] *= 2
        # energy spectrum = amplitude^2 / wl, double non-DC bins
        spec[h, :] = (winfft0[:hwl] ** 2) / wl
        spec[h, 1:] *= 2
        fft_avg += winfft0[:hwl]

    fft_avg /= n_win
    f = np.arange(wl) * (fs / wl)
    f = f[:hwl]
    return fft_avg, f, ffts, spec


def _q_from_spec(spec_1d, f, q=0.5):
    """
    Find frequency where cumulative energy first exceeds q * total.
    Mirrors MATLAB: Q = f(max(k-1, 1)) at first k where sum(spec(1:k)) > q*total.
    """
    total = float(np.sum(spec_1d))
    if total == 0.0:
        return 0.0
    cumsum = np.cumsum(spec_1d)
    idx = np.where(cumsum > q * total)[0]
    if len(idx) == 0:
        return float(f[-1])
    return float(f[max(idx[0] - 1, 0)])


def extract_schlegel21_parameters(wav_file_path, savepath):
    """
    Compute Schlegel21 parameters from raw amplitude-normalised audio.
    Uses the algorithms from FileS1.m (Linhart 2015 / Tallet 2013 / Garcia 2015)
    and FileS2 (Praat cross-correlation HNR).
    """
    from scipy.io import wavfile

    fs, audio_in = wavfile.read(str(wav_file_path))
    if audio_in.size == 0:
        raise ValueError("Empty audio file")

    if audio_in.ndim > 1:
        audio_in = np.mean(audio_in, axis=1).astype(np.float64)
    else:
        audio_in = audio_in.astype(np.float64)

    # Amplitude-normalise to [-1, 1] (matches MATLAB: y = y./max(abs(y)))
    max_abs = np.max(np.abs(audio_in))
    if max_abs > 0:
        audio_in = audio_in / max_abs

    L = len(audio_in)

    # ── Dur ──────────────────────────────────────────────────────────────────
    dur = L / fs

    # ── Tallet: whole-signal energy spectrum (one large window) ──────────────
    # windowedFFT(y, Fs, L, 0.875, 'hamming') → N_win=1
    _, f_whole, _, spec_whole = _windowed_fft(audio_in, fs, L, 0.875, 'hamming')
    spec_whole_1d = spec_whole[0]

    # PF: frequency bin with maximum energy
    pf = float(f_whole[np.argmax(spec_whole_1d)])

    # ── Tallet: 9 partial windows (windows 2..10 of 11) ─────────────────────
    wl_t = 1024
    n_win_t = 11
    # MATLAB: floor(1 : (L-wl+1)/(n_win-1) : (L-wl+1)), converted to 0-indexed
    _step = max(L - wl_t, 0) / (n_win_t - 1) if n_win_t > 1 else 0
    start_pos = np.floor(np.arange(n_win_t) * _step).astype(int)

    q50_windows = []
    for k in range(1, n_win_t - 1):   # k = 1..9  →  windows 2..10 of 11
        seg = audio_in[start_pos[k]: start_pos[k] + wl_t]
        if len(seg) < wl_t:
            seg = np.pad(seg, (0, wl_t - len(seg)))
        _, f9, _, spec9 = _windowed_fft(seg, fs, wl_t, 0, 'hamming')
        q50_windows.append(_q_from_spec(spec9[0], f9, 0.5))

    # Q50_2  : call-start Q50 (window 2 of 11, first of the 9 middle windows)
    q50_2 = float(q50_windows[0])
    # Q50_10 : call-end Q50 (window 10 of 11, last of the 9 middle windows)
    q50_10 = float(q50_windows[-1])
    # Q50_min: minimum Q50 across all 9 partial windows
    q50_min = float(np.min(q50_windows))

    # ── Garcia: per-frame Q25 (wl=1024, ovl=0.5), then averaged ─────────────
    _, f_g, _, spec_g = _windowed_fft(audio_in, fs, 1024, 0.5, 'hamming')
    q25 = float(np.mean([_q_from_spec(spec_g[k], f_g, 0.25)
                          for k in range(spec_g.shape[0])]))

    # ── HNR via parselmouth (Praat cross-correlation, FileS2) ─────────────────
    hnr = float('nan')
    try:
        import parselmouth
        sound = parselmouth.Sound(values=audio_in, sampling_frequency=float(fs))
        harmonicity = sound.to_harmonicity_cc(
            time_step=0.01,
            minimum_pitch=50.0,       # FileS2 uses 75.0 for humans → 50.0 for pigs
            silence_threshold=0.1,
            periods_per_window=1.0,
        )
        vals = harmonicity.values.flatten()
        # parselmouth uses -200.0 as Praat's sentinel for unvoiced/undefined frames
        voiced = vals[vals != -200.0]
        hnr = float(np.mean(voiced)) if len(voiced) > 0 else float('nan')
    except Exception:
        pass

    metrics = {
        'Schlegel21_Q50_2':   q50_2,
        'Schlegel21_Q50_10':  q50_10,
        'Schlegel21_Q50_min': q50_min,
        'Schlegel21_PF':      pf,
        'Schlegel21_Q25':     q25,
        'Schlegel21_Dur':     dur,
        'Schlegel21_HNR':     hnr,
    }
    df = pd.DataFrame([metrics])
    with open(savepath, 'wb') as f_out:
        pickle.dump(df, f_out)


# ============================================================================

def find_cycles_folder_for_squeal(wav_file, audio_snips_folder, cycles_base_folder):
    """
    Find cycles folder using the mirrored directory structure

    The Detected_Cycles folder now mirrors the Audio_Snips structure:

    Given WAV file:
      Audio_Snips/Bilateral/week_0_Beck and Kurt_B_Snip/ZOOM0007_LR-0001_sound_1020.wav

    Look for cycles in:
      Detected_Cycles/Bilateral/week_0_Beck and Kurt_B_Snip/ZOOM0007_LR-0001_sound_1020/

    Parameters:
    -----------
    wav_file : Path
        Path to the WAV file
    audio_snips_folder : Path
        Base Audio_Snips folder
    cycles_base_folder : Path
        Base Detected_Cycles folder

    Returns:
    --------
    Path or None : Path to cycles folder if found, None otherwise
    """
    audio_snips_folder = Path(audio_snips_folder)
    cycles_base_folder = Path(cycles_base_folder)

    if not cycles_base_folder.exists():
        return None

    # Get relative path of the wav file's parent folder from Audio_Snips base
    # e.g., "Bilateral/week_0_Beck and Kurt_B_Snip"
    try:
        rel_path = wav_file.parent.relative_to(audio_snips_folder)
    except ValueError:
        # wav_file is not under audio_snips_folder
        return None

    # Get squeal name (filename without .wav extension)
    squeal_name = wav_file.stem

    # Construct cycles folder path using mirrored structure
    cycles_folder = cycles_base_folder / rel_path / squeal_name

    if cycles_folder.exists() and cycles_folder.is_dir():
        return cycles_folder
    else:
        return None


def find_most_recent_cycles(cycles_folder):
    """
    Find the most recent .cycles file in the given folder

    Parameters:
    -----------
    cycles_folder : Path
        Path to folder containing .cycles files

    Returns:
    --------
    Path or None : Path to most recent .cycles file
    """
    if cycles_folder is None or not cycles_folder.exists():
        return None

    cycles_files = list(cycles_folder.glob("*.cycles"))

    if not cycles_files:
        return None

    # Extract timestamp from filename (format: name@DDMmmYY(HH-MM).cycles)
    def extract_timestamp(filepath):
        match = re.search(r'@(\d{2}[A-Za-z]{3}\d{2})\((\d{2})-(\d{2})\)', filepath.name)
        if match:
            date_str = match.group(1)
            hour = match.group(2)
            minute = match.group(3)
            try:
                dt = datetime.strptime(f"{date_str} {hour}:{minute}", "%d%b%y %H:%M")
                return dt
            except:
                pass
        return datetime.min

    # Return most recent file
    most_recent = max(cycles_files, key=extract_timestamp)
    return most_recent


def extract_features_for_audio(wav_file, audio_snips_folder, cycles_folder, output_base):
    """
    Extract all features for a single audio file

    Parameters:
    -----------
    wav_file : Path
        Path to WAV file
    audio_snips_folder : Path
        Base Audio_Snips folder
    cycles_folder : Path
        Base cycles folder
    output_base : Path
        Base output directory

    Returns:
    --------
    bool : True if successful (at least some features extracted), False otherwise
    """
    squeal_name = wav_file.stem  # Filename without .wav extension

    # Find cycles folder for this squeal using mirrored directory structure
    audio_cycles_folder = find_cycles_folder_for_squeal(wav_file, audio_snips_folder, cycles_folder)

    if not audio_cycles_folder:
        tqdm.write(f"  ✗ No cycles folder found for: {squeal_name}")
        return False

    # Find most recent cycles file
    cycle_file = find_most_recent_cycles(audio_cycles_folder)

    if not cycle_file:
        tqdm.write(f"  ✗ No cycles file found in: {audio_cycles_folder.name}")
        return False

    # Load audio and cycles
    try:
        vloader = VowelLoader()
        vloader.load(str(wav_file), str(cycle_file))
        wav, fs, cycles = vloader.wav, vloader.fs, vloader.cycles
    except Exception as e:
        tqdm.write(f"  ✗ Error loading files: {str(e)[:50]}...")
        return False

    # Preprocess
    wav = preprocess(wav, fs)

    # Extract metadata from WAV file path for structured output
    # Path format: Audio_Snips/Treatment/week_X_Subject_Type_Snip/squeal.wav
    try:
        rel_path = wav_file.parent.relative_to(audio_snips_folder)
        treatment_name = rel_path.parts[0]  # e.g., "Bilateral"
        week_folder = rel_path.parts[1]     # e.g., "week_0_Beck and Kurt_B_Snip"

        # Extract week number and subject from folder name
        import re
        match = re.match(r'week_(\d+)_(.+)_[BSU]_Snip', week_folder, re.IGNORECASE)
        if match:
            week_num = int(match.group(1))
            subject_name = match.group(2)
        else:
            # Fallback to flat structure if parsing fails
            audio_output = output_base / squeal_name
            audio_output.mkdir(parents=True, exist_ok=True)
            week_num = None
    except Exception:
        # Fallback to flat structure if metadata extraction fails
        audio_output = output_base / squeal_name
        audio_output.mkdir(parents=True, exist_ok=True)
        week_num = None

    # Create structured output folder: output/Treatment/Subject/week_X/squeal_name/
    if week_num is not None:
        audio_output = output_base / treatment_name / subject_name / f"week_{week_num}" / squeal_name
        audio_output.mkdir(parents=True, exist_ok=True)
    # else: already created flat structure above

    # Track feature extraction results
    feature_results = {}
    failed_features = []

    # Extract features with individual error handling
    # A - GAT features
    try:
        savepath = str(audio_output / 'a_gat.pickle')
        if not os.path.isfile(savepath):
            a_gat.extract(wav, fs, cycles, savepath)
        feature_results['a_gat'] = 'success'
    except Exception as e:
        feature_results['a_gat'] = 'failed'
        failed_features.append(('a_gat', str(e)))

    # B - DFG features (using 'f' as default for pigs)
    try:
        savepath = str(audio_output / 'b_dfg.pickle')
        if not os.path.isfile(savepath):
            b_dfg.extract(wav, fs, cycles, savepath)
        feature_results['b_dfg'] = 'success'
    except Exception as e:
        feature_results['b_dfg'] = 'failed'
        failed_features.append(('b_dfg', str(e)))

    # C - Temporal features
    try:
        savepath = str(audio_output / 'c_temporal.pickle')
        if not os.path.isfile(savepath):
            c_temporal.extract(wav, fs, cycles, savepath)
        feature_results['c_temporal'] = 'success'
    except Exception as e:
        feature_results['c_temporal'] = 'failed'
        failed_features.append(('c_temporal', str(e)))

    # D - Spectral features
    try:
        savepath = str(audio_output / 'd_spectral.pickle')
        if not os.path.isfile(savepath):
            d_spectral.extract(wav, fs, cycles, savepath)
        feature_results['d_spectral'] = 'success'
    except Exception as e:
        feature_results['d_spectral'] = 'failed'
        failed_features.append(('d_spectral', str(e)))

    # E1 - MFCC
    try:
        savepath = str(audio_output / 'e1_mfcc.pickle')
        if not os.path.isfile(savepath):
            e_cepstral.extract_mfcc(wav, fs, cycles, savepath)
        feature_results['e1_mfcc'] = 'success'
    except Exception as e:
        feature_results['e1_mfcc'] = 'failed'
        failed_features.append(('e1_mfcc', str(e)))

    # E2 - GFCC
    try:
        savepath = str(audio_output / 'e2_gfcc.pickle')
        if not os.path.isfile(savepath):
            e_cepstral.extract_gfcc(wav, fs, cycles, savepath)
        feature_results['e2_gfcc'] = 'success'
    except Exception as e:
        feature_results['e2_gfcc'] = 'failed'
        failed_features.append(('e2_gfcc', str(e)))

    # E3 - PLP
    try:
        savepath = str(audio_output / 'e3_plpcc.pickle')
        if not os.path.isfile(savepath):
            e_cepstral.extract_plp(wav, fs, cycles, savepath)
        feature_results['e3_plpcc'] = 'success'
    except Exception as e:
        feature_results['e3_plpcc'] = 'failed'
        failed_features.append(('e3_plpcc', str(e)))

    # E4 - RASTA-PLP (disabled: RASTA temporal filter is designed for long continuous
    # speech and distorts the fast spectral modulations that carry pig squeal information)
    # e_cepstral.extract_rasta(...) — not called

    # E5 - PNCC
    try:
        savepath = str(audio_output / 'e5_pncc.pickle')
        if not os.path.isfile(savepath):
            e_cepstral.extract_pncc(wav, fs, cycles, savepath)
        feature_results['e5_pncc'] = 'success'
    except Exception as e:
        feature_results['e5_pncc'] = 'failed'
        failed_features.append(('e5_pncc', str(e)))

    # F - LPC features
    try:
        savepath = str(audio_output / 'f_lpc.pickle')
        if not os.path.isfile(savepath):
            f_lpc.extract(wav, fs, cycles, savepath)
        feature_results['f_lpc'] = 'success'
    except Exception as e:
        feature_results['f_lpc'] = 'failed'
        failed_features.append(('f_lpc', str(e)))

    # G - AVCA features (MATLAB required)
    if MATLAB_AVAILABLE:
        # G1 - Perturbation set
        try:
            savepath = str(audio_output / 'g1_pert.pickle')
            if not os.path.isfile(savepath):
                g_avca.extract_avca_pert(wav, fs, cycles, savepath)
            feature_results['g1_pert'] = 'success'
        except Exception as e:
            feature_results['g1_pert'] = 'failed'
            failed_features.append(('g1_pert', str(e)))

        # G2 - Modulation spectrum set
        try:
            savepath = str(audio_output / 'g2_ms.pickle')
            if not os.path.isfile(savepath):
                g_avca.extract_avca_ms(wav, fs, cycles, savepath)
            feature_results['g2_ms'] = 'success'
        except Exception as e:
            feature_results['g2_ms'] = 'failed'
            failed_features.append(('g2_ms', str(e)))

        # G3 - Complexity set
        try:
            savepath = str(audio_output / 'g3_comp_fix.pickle')
            if not os.path.isfile(savepath):
                g_avca.extract_avca_comp(wav, fs, cycles, 4.0, 35.0, savepath)
            feature_results['g3_comp_fix'] = 'success'
        except Exception as e:
            feature_results['g3_comp_fix'] = 'failed'
            failed_features.append(('g3_comp_fix', str(e)))

    # H - IMF features (MATLAB required)
    if IMF_AVAILABLE:
        try:
            savepath = str(audio_output / 'h_imf.pickle')
            if not os.path.isfile(savepath):
                h_imf.extract(wav, fs, savepath)
            feature_results['h_imf'] = 'success'
        except Exception as e:
            feature_results['h_imf'] = 'failed'
            failed_features.append(('h_imf', str(e)))

    # I - Ratio features
    if RATIO_AVAILABLE:
        try:
            savepath = str(audio_output / 'i_ratio.pickle')
            if not os.path.isfile(savepath):
                i_ratio.extract(wav, fs, savepath)
            feature_results['i_ratio'] = 'success'
        except Exception as e:
            feature_results['i_ratio'] = 'failed'
            failed_features.append(('i_ratio', str(e)))

    # Z - UCLA parameters (F0, Jitter, Shimmer, Q50, Flux) — uses raw WAV file
    try:
        savepath = str(audio_output / 'z_ucla_parameters.pickle')
        if not os.path.isfile(savepath):
            extract_ucla_parameters(wav_file, savepath)
        feature_results['z_ucla_parameters'] = 'success'
    except Exception as e:
        feature_results['z_ucla_parameters'] = 'failed'
        failed_features.append(('z_ucla_parameters', str(e)))

    # Z2 - Schlegel21 parameters (Q50_2, Q50_10, Q50_min, PF, Q25, Dur, HNR) — uses raw WAV file
    try:
        savepath = str(audio_output / 'z2_schlegel21_parameters.pickle')
        if not os.path.isfile(savepath):
            extract_schlegel21_parameters(wav_file, savepath)
        feature_results['z2_schlegel21_parameters'] = 'success'
    except Exception as e:
        feature_results['z2_schlegel21_parameters'] = 'failed'
        failed_features.append(('z2_schlegel21_parameters', str(e)))

    # Report results
    success_count = sum(1 for v in feature_results.values() if v == 'success')
    total_count = len(feature_results)

    if failed_features:
        tqdm.write(f"  ⚠ Partial success: {success_count}/{total_count} features extracted")
        for feature_name, error in failed_features:
            tqdm.write(f"    - {feature_name} failed: {error[:50]}...")

    # Return True if at least some features succeeded
    return success_count > 0


def scan_and_process_all_files():
    """
    Scan all audio folders and process all .wav files found

    Returns:
    --------
    dict : Processing statistics
    """
    print("="*70)
    print("PIG AUDIO FEATURE EXTRACTION - DIRECTORY SCAN MODE")
    print("="*70)
    print()
    print(f"Audio folder: {AUDIO_SNIPS_FOLDER}")
    print(f"Cycles folder: {CYCLES_FOLDER}")
    print(f"Output folder: {OUTPUT_FOLDER}")
    print()

    # Check folders exist
    if not AUDIO_SNIPS_FOLDER.exists():
        print(f"ERROR: Audio folder not found: {AUDIO_SNIPS_FOLDER}")
        return None

    if not CYCLES_FOLDER.exists():
        print(f"ERROR: Cycles folder not found: {CYCLES_FOLDER}")
        return None

    # Create output folder
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

    # Collect all WAV files from all week folders
    all_wav_files = []

    print("Scanning for WAV files...")

    # Scan each treatment type folder
    for treatment_folder in AUDIO_SNIPS_FOLDER.iterdir():
        if not treatment_folder.is_dir():
            continue

        treatment_name = treatment_folder.name
        print(f"  Scanning {treatment_name}/...")

        # Scan each week folder
        for week_folder in treatment_folder.iterdir():
            if not week_folder.is_dir():
                continue

            # Find all .wav files in this folder
            wav_files = list(week_folder.glob("*.wav"))

            if wav_files:
                print(f"    {week_folder.name}: {len(wav_files)} files")
                all_wav_files.extend(wav_files)

    print()
    print(f"Total WAV files found: {len(all_wav_files)}")
    print()

    if not all_wav_files:
        print("No WAV files found!")
        return None

    # Process each file
    print("Processing files...")
    print("-"*70)

    stats = {
        'total': len(all_wav_files),
        'success': 0,
        'partial': 0,
        'failed': 0,
        'no_cycles': 0
    }

    for wav_file in tqdm(all_wav_files, desc="Extracting features"):
        tqdm.write(f"\nProcessing: {wav_file.name}")

        success = extract_features_for_audio(wav_file, AUDIO_SNIPS_FOLDER, CYCLES_FOLDER, OUTPUT_FOLDER)

        if success:
            stats['success'] += 1
            tqdm.write(f"  ✓ Success")
        else:
            stats['failed'] += 1
            tqdm.write(f"  ✗ Failed")

    print()
    print("-"*70)
    print("SUMMARY")
    print("-"*70)
    print(f"Total files:     {stats['total']}")
    print(f"Success:         {stats['success']} ({stats['success']/stats['total']*100:.1f}%)")
    print(f"Failed:          {stats['failed']} ({stats['failed']/stats['total']*100:.1f}%)")
    print()

    return stats


def main():
    """Main function"""
    print()
    print("="*70)
    print("STARTING FEATURE EXTRACTION (DIRECTORY SCAN MODE)")
    print("="*70)
    print()
    print("This script scans directories directly and processes ALL .wav files found.")
    print("It does not rely on Excel metadata.")
    print()

    stats = scan_and_process_all_files()

    if stats:
        print()
        print("="*70)
        print("EXTRACTION COMPLETE!")
        print("="*70)
        print()
        print(f"Results saved to: {OUTPUT_FOLDER}")


if __name__ == "__main__":
    main()
