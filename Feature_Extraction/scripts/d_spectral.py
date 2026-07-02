# from sustained_a.utils import tableio  # Not needed for extract() function
from svfel.core import SVFEx
from svfel.utils import util, tft
from svfel.features import spectral

import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from time import time


def extract(wav, fs, cycles, savepath):
    # frames
    framegen = util.FrameGenerator(wav, fs)   
    framegen.set(4096, 2048, 'samples', np.hanning) # ~93ms, 50% overlap at 44.1kHz
    frames = framegen.get_all()
    
    ### feature extraction
    features = []
    names = []
    
    # spectrogram
    NFFT = 4096
    S, xf = tft.rspectrogram(frames, fs, nfft=NFFT, power=1., norm=False)
    
    # spectral centroid
    centroid = spectral.spectral_centroid(S, fs, NFFT)
    
    features += [np.mean(centroid), np.std(centroid)]
    names += ['s_centroid_mean', 's_centroid_std']
    
    # spectral bandwidth
    bandwidth = spectral.spectral_bandwidth(S, fs, NFFT)  
    
    features += [np.mean(bandwidth), np.std(bandwidth)]
    names += ['s_bandwidth_mean', 's_bandwidth_std']
    
    # spectral rolloff
    rolloff = spectral.spectral_rolloff(S, fs, NFFT, roll_p=0.85)
    
    features += [np.mean(rolloff), np.std(rolloff)]
    names += ['s_rolloff_mean', 's_rolloff_std']
    
    # spectral rollon
    rollon = spectral.spectral_rolloff(S, fs, NFFT, roll_p=0.15)

    features += [np.mean(rollon), np.std(rollon)]
    names += ['s_rollon_mean', 's_rollon_std']
    
    # spectral contrast
    contrast, fbands = spectral.spectral_contrast(S, fs, NFFT, fmin=50, nbands=7, quantile=0.02, log=True)
    for n, c in enumerate(contrast.T):
        features += [np.mean(c), np.std(c)]
        names += [f's_contrast_band_{n}_mean', f's_contrast_band_{n}_std']
    
    # spectral flatness
    flatness = spectral.spectral_flatness(S, amin=1e-10, power=2.0)

    features += [np.mean(flatness), np.std(flatness)]
    names += ['s_flatness_mean', 's_flatness_std']

    # spectral flux
    flux  = spectral.spectral_flux(S, eps=1e-12, norm=True, p=2.0)

    features += [np.mean(flux), np.std(flux)]
    names += ['s_flux_mean', 's_flux_std']
    
    # spectral entropy
    entropy = spectral.spectral_entropy(S)
     
    features += [np.mean(entropy), np.std(entropy)]
    names += ['s_entropy_mean', 's_entropy_std']
    
    # log spectral distance
    S_norm = S / np.sum(S, axis=-1, keepdims=True)
    dist = spectral.log_spectral_distance(S_norm[:-1], S_norm[1:])
    
    features += [np.mean(dist), np.std(dist)]
    names += ['s_logdist_mean', 's_logdist_std']
    
    # to df
    features = [np.squeeze(f) for f in features]
    features = np.array([features])
    df = pd.DataFrame(features, columns=names)
    
    # save
    os.makedirs(os.path.dirname(savepath), exist_ok=True)
    with open(savepath, 'wb') as f:
        pickle.dump(df, f)




