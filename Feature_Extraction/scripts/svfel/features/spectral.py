# https://github.com/librosa/librosa/blob/main/librosa/feature/spectral.py
# https://musicinformationretrieval.com/spectral_features.html
# https://github.com/fraunhoferportugal/tsfel/blob/21b023aae69fa1a1ff9abe3c534980fa7883025f/tsfel/feature_extraction/features.py


from svfel.utils import util

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import periodogram, welch

def spectral_centroid(S, fs, nfft):
    # Each frame of a magnitude spectrogram is normalized and treated as a
    # distribution over frequency bins, from which the mean (centroid) is
    # extracted per frame
    # https://github.com/librosa/librosa/blob/main/librosa/feature/spectral.py
    # https://musicinformationretrieval.com/spectral_features.html
    if np.any(S < 0):
        raise ValueError("S must have non-negative energies only")
    # get frequency bins
    # if freqs is None:
    freqs = np.fft.rfftfreq(nfft, d=1./fs).reshape(1,-1)
    # get spectral centroid
    c = np.sum(S*freqs, axis=-1, keepdims=True) / np.sum(S, axis=-1, keepdims=True)
    return c
    

def spectral_bandwidth(S, fs, nfft, p=2, fc=None):
    # Compute p'th-order spectral bandwidth.
    # https://github.com/librosa/librosa/blob/main/librosa/feature/spectral.py
    # https://musicinformationretrieval.com/spectral_features.html
    if np.any(S < 0):
        raise ValueError("S must have non-negative energies only")
    # get frequency bins
    freqs = np.fft.rfftfreq(nfft, d=1./fs).reshape(1,-1)
    # get spectral centroid
    if fc is None:
        fc = spectral_centroid(S, fs, nfft)
    # get spectral bandwidth
    bw = (np.sum(S * (freqs - fc)**p, axis=-1, keepdims=True) / np.sum(S, axis=-1, keepdims=True))**(1./p) 
    return bw
    

def spectral_contrast(S, fs, nfft, fmin=200., nbands=6, quantile=0.02, log=True):
    # Each frame of a spectrogram ``S`` is divided into sub-bands.
    # For each sub-band, the energy contrast is estimated by comparing
    # the mean energy in the top quantile (peak energy) to that of the
    # bottom quantile (valley energy).  High contrast values generally
    # correspond to clear, narrow-band signals, while low contrast values
    # correspond to broad-band noise.
    # bands = ocatves of fmin (doubles/halves)
    # https://github.com/librosa/librosa/blob/main/librosa/feature/spectral.py
    # https://musicinformationretrieval.com/spectral_features.html
    
    # get frequency bins
    freqs = np.fft.rfftfreq(nfft, d=1./fs).reshape(1,-1)
    # create frequency bands
    f_band = np.zeros(nbands + 2)
    f_band[1:] = fmin * (2.0 ** np.arange(0, nbands + 1))
    if np.any(f_band[:-1] >= 0.5 * fs):
        raise ValueError("Frequency band exceeds Nyquist. Reduce either fmin or n_bands.")
    
    valley, peak = [],[]
    
    # for each band
    for k, (f_low, f_high) in enumerate(zip(f_band[:-1], f_band[1:])):
        # get corresponding bins
        current_band = np.logical_and(freqs >= f_low, freqs <= f_high).squeeze() # mask
        idx = np.flatnonzero(current_band)                                       # idx

        # indexes are adjusted slightly...not wrong but why?
        if k > 0:
            current_band[idx[0] - 1] = True

        if k == nbands:
            current_band[idx[-1] + 1 :] = True
        
        # get magnitudes
        sub_band = S[:, current_band]

        if k < nbands:
            sub_band = sub_band[:, :-1]

        # get number of bins in quantile
        nq = np.rint(quantile * np.sum(current_band))
        # always take at least one bin from each side if quantile * n <= 0.5
        nq = int(np.maximum(nq, 1))
        
        # sort magnitudes
        sortedr = np.sort(sub_band, axis=-1)
        
        # nq smallest magnitudes
        valley.append(np.mean(sortedr[:, :nq], axis=-1))
        # nq biggest magnitudes
        peak.append(np.mean(sortedr[:, -nq:], axis=-1))
    
    # stack
    valley = np.stack(valley).T
    peak = np.stack(peak).T

    if log:
        return util.amp_to_db(peak) - util.amp_to_db(valley), f_band
        # librosa uses power to db even though power=1??? -> no qualitative changes, just factor 2
        # return utils.pow_to_db(peak) - utils.pow_to_db(valley), f_band
    else:
        return peak - valley, f_band
        
    
def spectral_rolloff(S, fs, nfft, roll_p=0.85):
    # The roll-off frequency is defined for each frame as the center frequency
    # for a spectrogram bin such that at least roll_percent (0.85 by default)
    # of the energy of the spectrum in this frame is contained in this bin and
    # the bins below. This can be used to, e.g., approximate the maximum (or
    # minimum) frequency by setting roll_percent to a value close to 1 (or 0).
    # https://github.com/librosa/librosa/blob/main/librosa/feature/spectral.py
    # https://musicinformationretrieval.com/spectral_features.html
    if np.any(S < 0):
        raise ValueError("S must have non-negative energies only")
    # get frequency bins
    freqs = np.fft.rfftfreq(nfft, d=1./fs).reshape(1,-1)
    # cumulative energy/amplitude
    total_energy = np.cumsum(S, axis=-1)
    # threshold = roll_p * max
    threshold = (roll_p * total_energy[:, -1]).reshape(-1, 1)
    # get first frequency where cumulative energy >= threshold
    ind = np.where(total_energy < threshold, np.nan, 1)
    rolloff = np.nanmin(ind * freqs, axis=-1, keepdims=True)
    return rolloff
    
    
def spectral_flatness(S, amin=1e-10, power=2.0):
    # Spectral flatness (or tonality coefficient) is a measure to
    # quantify how much noise-like a sound is, as opposed to being
    # tone-like [#]_. A high spectral flatness (closer to 1.0)
    # indicates the spectrum is similar to white noise.
    # It is often converted to decibel.
    # https://github.com/librosa/librosa/blob/main/librosa/feature/spectral.py
    # https://musicinformationretrieval.com/spectral_features.html
    if np.any(S < 0):
        raise ValueError("S must have non-negative energies only")
    # minimum value
    S_thresh = np.maximum(amin, S ** power)
    # geometric mean
    gmean = np.exp(np.mean(np.log(S_thresh), axis=-1, keepdims=True))
    # arithmetic mean
    amean = np.mean(S_thresh, axis=-1, keepdims=True)
    return gmean / amean
 
  
def spectral_entropy(S):
    # shannon entropy of power spectrum
    # power spectrum
    P = S ** 2
    # normalize
    P_norm = P / np.sum(P, axis=-1, keepdims=True)
    # return spectral entropy
    return -np.nansum(util.xlogx(P_norm, 2), axis=-1, keepdims=True) / np.log2(P_norm.shape[-1])

#### NEW FEATURE: SPECTRAL FLUX - how quickly spectrum changes over time. computes p-norm b/w consecutive frames

def spectral_flux(S, eps=1e-12, norm=True, p=2.0):
    if np.any(S < 0):
        raise ValueError("S must have non-negative energies only")

    if S.shape[0] < 2:
        return np.zeros((0, 1))
    
    X = S.astype(float)
    
    if norm:
        X = X / (np.sum(X, axis=-1, keepdims=True) + eps)

    diff = X[1:] - X[:-1]

    flux = np.sum(np.abs(diff) ** p, axis=-1, keepdims=True)
    
    flux = flux ** (1.0 / p)
    
    return flux


# for lpc envelope?
def log_spectral_distance(S, S_hat):
    # https://en.wikipedia.org/wiki/Log-spectral_distance
    return np.sqrt(np.sum(util.amp_to_db(S**2 / S_hat**2)**2, axis=-1, keepdims=True)) / S.shape[-1]


def chroma_stft(
    sr=22050,
    S=None,
    norm=np.inf,
    n_fft=2048,
    tuning=None,
    n_chroma=12,
):
    import librosa

    if tuning is None:
        tuning = librosa.core.pitch.estimate_tuning(S=S, sr=sr, bins_per_octave=n_chroma)

    # Get the filter bank
    chromafb = librosa.filters.chroma(
        sr=sr, n_fft=n_fft, tuning=tuning, n_chroma=n_chroma,
    )

    # Compute raw chroma
    raw_chroma = np.einsum("cf,...ft->...ct", chromafb, S, optimize=True)

    # Compute normalization factor for each frame
    return librosa.util.normalize(raw_chroma, norm=norm, axis=-2)
