from svfel.utils import util
from svfel.utils import tft, fbanks

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import periodogram, welch

from svfel.utils import util
from svfel.features import cepstral

import os
import pickle
import numpy as np
import pandas as pd


def extract(wav, fs, savepath):
    # frames
    framegen = util.FrameGenerator(wav, fs) 
    framegen.set(4096, 2048, 'samples', np.hanning) # ~93ms, 50% overlap
    frames = framegen.get_all()

    S, xf = tft.rspectrogram(frames, fs, nfft=4096, power=1., norm=False)
    
    ### feature extraction
    features = []
    names = []
    
    # low-to-high ratio
    lhr = low_high_ratio(S, fs, nfft=4096)
    features += [np.mean(lhr), np.std(lhr)]
    names += ['lhr_mean', 'lhr_std']
    
    # soft phonation index
    spi = soft_phonation_index_mel(S, fs, nfft=4096)
    features += [np.mean(spi), np.std(spi)]
    names += ['spi_mean', 'spi_std']
    
    # to df
    features = [np.squeeze(f) for f in features]
    features = np.array([features])
    df = pd.DataFrame(features, columns=names)
    
    # save
    os.makedirs(os.path.dirname(savepath), exist_ok=True)
    with open(savepath, 'wb') as f:
        pickle.dump(df, f)
              
        
def low_high_ratio(S, fs, nfft):
    # ratio between low & high frequency band (4000Hz) from AVCA/Awan 
    # get frequency bins
    freqs = np.fft.rfftfreq(nfft, d=1./fs).reshape(1,-1)
    # find 4000Hz bin
    idx = tft.get_bin(freqs, 4000)
    # get band energy
    mean_mag_low = S[:, :idx].sum(axis=1, keepdims=True)
    mean_mag_high = S[:, idx:].sum(axis=1, keepdims=True)
    eps = 1e-12
    # ratio
    lhr = (mean_mag_low + eps) / (mean_mag_high + eps)
    # return in db
    return util.pow_to_db(lhr)
    

def soft_phonation_index_mel(S, fs, nfft):
    # ratio between 70-1600Hz & 1600-4500Hz bands from Tulics 2019
    # filterbank 
    fbank, fc = fbanks.mel_filterbank(fs, nfft, nfilt=22, fmin=50, fmax=10000, debug=False)
    # apply mel filterbank
    S_mel = fbanks.apply(S, fbank)
    # get band energy (split at filter 11 ≈ 1600 Hz with new 50–10000 Hz range)
    S_mel_low = S_mel[:,:11].sum(axis=1, keepdims=True)
    S_mel_high = S_mel[:,11:].sum(axis=1, keepdims=True)
    eps = 1e-12
    # ratio
    ratio = (S_mel_low + eps) / (S_mel_high + eps)
    # return in db
    return util.pow_to_db(ratio)

