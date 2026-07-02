from svfel.utils import util
from svfel.features import temporal

import os
import pickle
import numpy as np
import pandas as pd


def extract(wav, fs, cycles, savepath):
    # Safety: clip cycle boundaries to signal length
    cycles = np.clip(cycles, 0, len(wav) - 1)

    # cycles
    cframes = util.cycle_split(wav, cycles)
    
    ### pitch
    T, F = temporal.cpitch(cframes, fs) # cycle values
    # Fc = util.spread_cycle_values(F, cycles, len(wav)) # spread to signal
    ### p2p amplitude
    A = temporal.cp2p_amplitude(cframes)       
    ### energy
    E = temporal.ctotal_energy(cframes, fs)
    ## FFT
    # S, xf = tft.rfft(wav.reshape(1,-1), fs, nfft=2**util.nextpow2(len(wav)), norm=True)
    ### zero crossing rate
    zcr = temporal.zcr(cframes)
    
    ### feature extraction
    features = []
    names = []

    features += [np.mean(A), np.std(A),
                 np.mean(E), np.std(E),
                 np.mean(zcr), np.std(zcr)]
    names += ['a_mean', 'a_std',
              'e_mean', 'e_std',
              'zcr_mean', 'zcr_std']
    
    # to df
    features = [np.squeeze(f) for f in features]
    features = np.array([features])
    df = pd.DataFrame(features, columns=names)
    
    # save
    os.makedirs(os.path.dirname(savepath), exist_ok=True)
    with open(savepath, 'wb') as f:
        pickle.dump(df, f)







