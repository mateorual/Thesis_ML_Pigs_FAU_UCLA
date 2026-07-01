# from sustained_a.utils import tableio  # Not needed for extract() function
from svfel.core import SVFEx
from svfel.utils import util, tft
from svfel.features import temporal
from svfel.features.dfg import cpx

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
    Fc = util.spread_cycle_values(F, cycles, len(wav)) # spread to signal
    
    ## FFT
    #S, xf = tft.rfft(wav.reshape(1,-1), fs, nfft=2**util.nextpow2(len(wav)), norm=True)
    
    ### feature extraction
    features = []
    names = []
    
    ### CPX
    framegen = util.FrameGenerator(wav, fs)
    # cpp, cpm
    framegen.set(4096, 2048, 'samples', np.hanning)
    frames = framegen.get_all()
    f0 = np.mean(framegen.split(Fc), axis=-1)
    cpp, fbins = cpx.cepstral_peak_x(frames, fs, f0, nfft=4096, n_smooth_t=0, n_smooth_q=0, debug=False)
    cpm, fbins = cpx.cepstral_peak_x(frames, fs, f0=None, nfft=4096, n_smooth_t=0, n_smooth_q=0, debug=False)
    # cpps, cpms
    framegen.set(4096, 256, 'samples', np.hanning) # experimenting with different values is recommended.
    frames = framegen.get_all()
    f0 = np.mean(framegen.split(Fc), axis=-1)
    cpps, fbins = cpx.cepstral_peak_x(frames, fs, f0, nfft=4096, n_smooth_t=150, n_smooth_q=3, debug=False)
    cpms, fbins = cpx.cepstral_peak_x(frames, fs, f0=None, nfft=4096, n_smooth_t=150, n_smooth_q=3, debug=False)
    # stats
    features += [np.mean(cpp), np.std(cpp),
                 np.mean(cpm), np.std(cpm),
                 np.mean(cpps), np.std(cpps),
                 np.mean(cpms), np.std(cpms)]
    names += ['cpp_mean', 'cpp_std',
              'cpm_mean', 'cpm_std',
              'cpps_mean', 'cpps_std',
              'cpms_mean', 'cpms_std'] 
    
    # to df
    features = [np.squeeze(f) for f in features]
    features = np.array([features])
    df = pd.DataFrame(features, columns=names)
    
    # save
    os.makedirs(os.path.dirname(savepath), exist_ok=True)
    with open(savepath, 'wb') as f:
        pickle.dump(df, f)


