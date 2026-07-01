from svfel.utils import util
from svfel.features import lpc

import os
import pickle
import numpy as np
import pandas as pd


def extract(wav, fs, cycles, savepath):
    # Safety: clip cycle boundaries to signal length
    cycles = np.clip(cycles, 0, len(wav) - 1)

    # Pig-specific values: 8 kHz captures vocal tract resonances while
    # keeping LPC order numerically stable on short frames
    cutoff = 8000
    fs_re = 16000
    order = lpc.get_order(6)  # 2 + 6*2 = 14 poles
    
    # filter signal
    wav_re = util.fir_lowpass(wav.reshape(1,-1), fs, cutoff=cutoff, window='hamming').squeeze()      
    # resample signal
    wav_re = util.resample_polyphase(wav_re, fs, fs_re)

    # frames
    framegen = util.FrameGenerator(wav_re, fs_re)
    framegen.set(1024, 512, 'samples', np.hanning) # ~93ms, 50% overlap
    frames = framegen.get_all()
    
    ### feature extraction
    features = []
    names = []

    # linear prediction coefficients
    a, e = lpc.lpc(frames, order=order)
    # mean/std lpc
    features += [np.mean(a[:,1:], axis=0)]
    features += [np.std(a[:,1:], axis=0)]
    names += [f'lpc_{i}_mean' for i in range(a[:,1:].shape[1])]
    names += [f'lpc_{i}_std' for i in range(a[:,1:].shape[1])]
    
    # line spectral pairs/frequencies
    lsf = util.rad_to_hz(lpc.lsp(a), fs_re)
    # mean/std lsf
    features += [np.mean(lsf, axis=0)]
    features += [np.std(lsf, axis=0)]
    names += [f'lsf_{i}_mean' for i in range(lsf.shape[1])]
    names += [f'lsf_{i}_std' for i in range(lsf.shape[1])]
    
    # linear prediction cepstral coefficients
    lpcc = lpc.lpcc(a, e)
    # mean/std lpcc
    features += [np.mean(lpcc, axis=0)]
    features += [np.std(lpcc, axis=0)]
    names += [f'lpcc_{i}_mean' for i in range(lpcc.shape[1])]
    names += [f'lpcc_{i}_std' for i in range(lpcc.shape[1])]
    
    
    # to df
    features = np.hstack(features).reshape(1,-1)
    df = pd.DataFrame(features, columns=names)

    # save
    os.makedirs(os.path.dirname(savepath), exist_ok=True)
    with open(savepath, 'wb') as f:
        pickle.dump(df, f)