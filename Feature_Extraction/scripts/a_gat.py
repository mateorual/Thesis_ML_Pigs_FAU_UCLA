# from utils import tableio  # Not needed for extract() function
from svfel.utils import util, tft
from svfel.features import temporal
from svfel.features.dfg import perturbation, noise, cpx

import os
import pickle
import numpy as np
import pandas as pd


def extract(wav, fs, cycles, savepath):
    # Safety: clip cycle boundaries to signal length
    # This prevents "index out of bounds" errors when cycles extend beyond signal
    cycles = np.clip(cycles, 0, len(wav) - 1)

    # get cycles
    cframes = util.cycle_split(wav, cycles)
    
    ### pitch
    T, F = temporal.cpitch(cframes, fs)
    ### p2p amplitude
    A = temporal.cp2p_amplitude(cframes)
    ### energy
    E = temporal.ctotal_energy(cframes, fs)
    ## FFT
    S, xf = tft.rfft(wav.reshape(1,-1), fs, nfft=2**util.nextpow2(len(wav)), norm=True)
        
    ### feature extraction
    features = []
    names = []
    
    ### f0
    features += [np.mean(F, axis=0), 
                 np.std(F, axis=0), 
                 np.max(F, axis=0), 
                 np.min(F, axis=0)]
    names += ['f0_mean', 'f0_std', 'f0_max', 'f0_min']
    
    ### period perturbation
    features += [perturbation.mean_jit(T.T),
                 perturbation.jit_percent(T.T),
                 perturbation.jit_factor(T.T),
                 perturbation.jit_ratio(T.T),
                 perturbation.perturbation_factor(T.T),
                 perturbation.pvi(T.T),
                 perturbation.perturbation_quotient(T.T, 3),
                 perturbation.perturbation_quotient(T.T, 5),
                 perturbation.perturbation_quotient(T.T, 11),
                 perturbation._rap(T.T),
                 perturbation._rap(T.T, v=2)]
    names += ['mean_jit', 'jit_p', 'jit_factor', 'jit_ratio', 'ppf', 'pvi', 'ppq3', 'ppq5', 'ppq11', 'rap_v1', 'rap_v2']

    ### amplitude perturbation
    features += [perturbation.mean_shim(A.T),
                 perturbation.shim_percent(A.T),
                 perturbation.perturbation_factor(A.T),
                 perturbation.avi(A.T),
                 perturbation.perturbation_quotient(A.T, 3),
                 perturbation.perturbation_quotient(A.T, 5),
                 perturbation.perturbation_quotient(A.T, 11)]
    names += ['mean_shim', 'shim_p', 'apf', 'avi', 'apq3', 'apq5', 'apq11']
        
    ### energy perturbation
    features += [perturbation.perturbation_factor(E.T),
                 perturbation.perturbation_quotient(E.T, 3),
                 perturbation.perturbation_quotient(E.T, 5),
                 perturbation.perturbation_quotient(E.T, 11)]
    names += ['epf', 'epq3', 'epq5', 'epq11']
    
    ### noise measures
    wmc = noise.wmc(cframes, wav) 
    snrv1 = noise.snr_v1_gat(wav, cycles, fs)
    features += [noise.harmonics_intensity(S, xf, F.mean(), f_tol=20, pthres=0.05),
                 noise.hnr(cframes),
                 noise.spectral_flatness_gat(S),
                 np.mean(wmc, axis=0),
                 np.max(wmc, axis=0),
                 cpx.cepstral_peak_x(wav.reshape(1,-1), fs, [F.mean()], nfft=2**util.nextpow2(wav.shape[-1]), n_smooth_t=0, n_smooth_q=0)[0],
                 noise.nne_gat(wav, cycles, fs),
                 np.mean(snrv1),
                 np.std(snrv1),
                 np.min(snrv1),
                 np.max(snrv1)]
    names += ['harmonics_intensity', 'hnr', 'spectral_flatness', 
              'wmc_mean', 'wmc_max', 'cpp', 'nne', 
              'snr1_mean', 'snr1_std', 'snr1_min', 'snr1_max']

    # missing: SNRv2, GNE
    
    # to df
    features = [np.squeeze(f) for f in features]
    features = np.array([features])
    df = pd.DataFrame(features, columns=names)

    # save
    os.makedirs(os.path.dirname(savepath), exist_ok=True)
    with open(savepath, 'wb') as f:
        pickle.dump(df, f)







