from svfel.utils import util
from svfel.features import cepstral

import os
import pickle
import numpy as np
import pandas as pd


def extract_mfcc(wav, fs, cycles, savepath):
    # Safety: clip cycle boundaries to signal length
    cycles = np.clip(cycles, 0, len(wav) - 1)

    # frames
    framegen = util.FrameGenerator(wav, fs)   
    framegen.set(4096, 2048, 'samples', np.hanning) # ~93ms, 50% overlap
    frames = framegen.get_all()
    
    ### feature extraction
    features = []
    names = []

    
    # mel frequency cepstral coefficients
    mfcc, _ = cepstral.mfcc(frames, fs, nfft=4096, nmfcc=12, nfilt=40, lifter=23,
                            fmin=50.0, fmax=16000.0, s_norm=True, debug=False)
    # delta mel frequency cepstral coefficients
    dmfcc = cepstral.delta_cc(mfcc, 1)
    # delta delta mel frequency cepstral coefficients
    ddmfcc = cepstral.delta_cc(mfcc, 2)
    
    # mean/std mfcc
    features += [np.mean(mfcc, axis=0)]
    features += [np.std(mfcc, axis=0)]
    names += [f'mfcc_{i}_mean' for i in range(mfcc.shape[1])]
    names += [f'mfcc_{i}_std' for i in range(mfcc.shape[1])]
    
    # mean/std dmfcc
    features += [np.mean(dmfcc, axis=0)]
    features += [np.std(dmfcc, axis=0)]
    names += [f'dmfcc_{i}_mean' for i in range(dmfcc.shape[1])]
    names += [f'dmfcc_{i}_std' for i in range(dmfcc.shape[1])]
    
    # mean/std ddmfcc
    features += [np.mean(ddmfcc, axis=0)]
    features += [np.std(ddmfcc, axis=0)]
    names += [f'ddmfcc_{i}_mean' for i in range(ddmfcc.shape[1])]
    names += [f'ddmfcc_{i}_std' for i in range(ddmfcc.shape[1])]

    # to df
    features = np.hstack(features).reshape(1,-1)
    df = pd.DataFrame(features, columns=names)

    # save
    os.makedirs(os.path.dirname(savepath), exist_ok=True)
    with open(savepath, 'wb') as f:
        pickle.dump(df, f)


def extract_gfcc(wav, fs, cycles, savepath):
    # Safety: clip cycle boundaries to signal length
    cycles = np.clip(cycles, 0, len(wav) - 1)

    # frames
    framegen = util.FrameGenerator(wav, fs)   
    framegen.set(4096, 2048, 'samples', np.hanning) # ~93ms, 50% overlap
    frames = framegen.get_all()
    
    ### feature extraction
    features = []
    names = []

    
    # gammatone frequency cepstral coefficients
    gfcc, _ = cepstral.gfcc(frames, fs, nfft=4096, ngfcc=12, nfilt=40, lifter=23,
                            fmin=50.0, fmax=16000.0, s_norm=True, debug=False)
    # delta gammatone frequency cepstral coefficients
    dgfcc = cepstral.delta_cc(gfcc, 1)
    # delta delta gammatone frequency cepstral coefficients
    ddgfcc = cepstral.delta_cc(gfcc, 2)
    
    # mean/std gfcc
    features += [np.mean(gfcc, axis=0)]
    features += [np.std(gfcc, axis=0)]
    names += [f'gfcc_{i}_mean' for i in range(gfcc.shape[1])]
    names += [f'gfcc_{i}_std' for i in range(gfcc.shape[1])]
    
    # mean/std dgfcc
    features += [np.mean(dgfcc, axis=0)]
    features += [np.std(dgfcc, axis=0)]
    names += [f'dgfcc_{i}_mean' for i in range(dgfcc.shape[1])]
    names += [f'dgfcc_{i}_std' for i in range(dgfcc.shape[1])]
    
    # mean/std ddgfcc
    features += [np.mean(ddgfcc, axis=0)]
    features += [np.std(ddgfcc, axis=0)]
    names += [f'ddgfcc_{i}_mean' for i in range(ddgfcc.shape[1])]
    names += [f'ddgfcc_{i}_std' for i in range(ddgfcc.shape[1])]

    # to df
    features = np.hstack(features).reshape(1,-1)
    df = pd.DataFrame(features, columns=names)

    # save
    os.makedirs(os.path.dirname(savepath), exist_ok=True)
    with open(savepath, 'wb') as f:
        pickle.dump(df, f)


def extract_plp(wav, fs, cycles, savepath):
    # Safety: clip cycle boundaries to signal length
    cycles = np.clip(cycles, 0, len(wav) - 1)

    # frames
    framegen = util.FrameGenerator(wav, fs)   
    framegen.set(4096, 2048, 'samples', np.hanning) # ~93ms, 50% overlap
    frames = framegen.get_all()
    
    ### feature extraction
    features = []
    names = []

    
    # perceptual linear prediction cepstral coeffcients
    a_plp, e_plp, _ = cepstral.plp(frames, fs, nfft=4096, lpc_order=12, nfilts=None, use_rasta=False,
                                   width=1., fmin=50.0, fmax=16000.0, debug=False)
    plpcc = cepstral.plpcc(a_plp, e_plp, lifter=23)           
    # delta plpcc
    dplpcc = cepstral.delta_cc(plpcc, 1)
    # delta delta plpcc
    ddplpcc = cepstral.delta_cc(plpcc, 2)
    
    # mean/std plpcc
    features += [np.mean(plpcc, axis=0)]
    features += [np.std(plpcc, axis=0)]
    names += [f'plpcc_{i}_mean' for i in range(plpcc.shape[1])]
    names += [f'plpcc_{i}_std' for i in range(plpcc.shape[1])]
    
    # mean/std dplpcc
    features += [np.mean(dplpcc, axis=0)]
    features += [np.std(dplpcc, axis=0)]
    names += [f'dplpcc_{i}_mean' for i in range(dplpcc.shape[1])]
    names += [f'dplpcc_{i}_std' for i in range(dplpcc.shape[1])]
    
    # mean/std ddplpcc
    features += [np.mean(ddplpcc, axis=0)]
    features += [np.std(ddplpcc, axis=0)]
    names += [f'ddplpcc_{i}_mean' for i in range(ddplpcc.shape[1])]
    names += [f'ddplpcc_{i}_std' for i in range(ddplpcc.shape[1])]

    # to df
    features = np.hstack(features).reshape(1,-1)
    df = pd.DataFrame(features, columns=names)

    # save
    os.makedirs(os.path.dirname(savepath), exist_ok=True)
    with open(savepath, 'wb') as f:
        pickle.dump(df, f)


def extract_rasta(wav, fs, cycles, savepath):
    # Safety: clip cycle boundaries to signal length
    cycles = np.clip(cycles, 0, len(wav) - 1)

    # frames
    framegen = util.FrameGenerator(wav, fs)   
    framegen.set(4096, 2048, 'samples', np.hanning) # ~93ms, 50% overlap
    frames = framegen.get_all()
    
    ### feature extraction
    features = []
    names = []

    # rasta perceptual linear prediction cepstral coeffcients
    a_rplp, e_rplp, _ = cepstral.plp(frames, fs, nfft=4096, lpc_order=12, nfilts=None, use_rasta=True,
                                     width=1., fmin=50.0, fmax=16000.0, debug=False)
    rastacc = cepstral.plpcc(a_rplp, e_rplp, lifter=23)        
    # delta rastacc
    drastacc = cepstral.delta_cc(rastacc, 1)
    # delta delta rastacc
    ddrastacc = cepstral.delta_cc(rastacc, 2)
    
    # mean/std rastacc
    features += [np.mean(rastacc, axis=0)]
    features += [np.std(rastacc, axis=0)]
    names += [f'rastacc_{i}_mean' for i in range(rastacc.shape[1])]
    names += [f'rastacc_{i}_std' for i in range(rastacc.shape[1])]
    
    # mean/std drastacc
    features += [np.mean(drastacc, axis=0)]
    features += [np.std(drastacc, axis=0)]
    names += [f'drastacc_{i}_mean' for i in range(drastacc.shape[1])]
    names += [f'drastacc_{i}_std' for i in range(drastacc.shape[1])]
    
    # mean/std ddrastacc
    features += [np.mean(ddrastacc, axis=0)]
    features += [np.std(ddrastacc, axis=0)]
    names += [f'ddrastacc_{i}_mean' for i in range(ddrastacc.shape[1])]
    names += [f'ddrastacc_{i}_std' for i in range(ddrastacc.shape[1])]

    # to df
    features = np.hstack(features).reshape(1,-1)
    df = pd.DataFrame(features, columns=names)

    # save
    os.makedirs(os.path.dirname(savepath), exist_ok=True)
    with open(savepath, 'wb') as f:
        pickle.dump(df, f)


def extract_pncc(wav, fs, cycles, savepath):
    # Safety: clip cycle boundaries to signal length
    cycles = np.clip(cycles, 0, len(wav) - 1)

    # frames
    framegen = util.FrameGenerator(wav, fs)   
    framegen.set(4096, 2048, 'samples', np.hanning) # ~93ms, 50% overlap
    frames = framegen.get_all()
    
    ### feature extraction
    features = []
    names = []

    
    # power normalized cepstral coefficients
    pncc, _ = cepstral.pncc(frames, fs, nfft=4096, nceps=12, nfilt=40, simple=False,
                            fmin=50.0, fmax=16000.0, debug=False)
    # delta pncc
    dpncc = cepstral.delta_cc(pncc, 1)
    # delta delta pncc
    ddpncc = cepstral.delta_cc(pncc, 2)
    
    # mean/std pncc
    features += [np.mean(pncc, axis=0)]
    features += [np.std(pncc, axis=0)]
    names += [f'pncc_{i}_mean' for i in range(pncc.shape[1])]
    names += [f'pncc_{i}_std' for i in range(pncc.shape[1])]
    
    # mean/std dpncc
    features += [np.mean(dpncc, axis=0)]
    features += [np.std(dpncc, axis=0)]
    names += [f'dpncc_{i}_mean' for i in range(dpncc.shape[1])]
    names += [f'dpncc_{i}_std' for i in range(dpncc.shape[1])]
    
    # mean/std ddpncc
    features += [np.mean(ddpncc, axis=0)]
    features += [np.std(ddpncc, axis=0)]
    names += [f'ddpncc_{i}_mean' for i in range(ddpncc.shape[1])]
    names += [f'ddpncc_{i}_std' for i in range(ddpncc.shape[1])]

    # to df
    features = np.hstack(features).reshape(1,-1)
    df = pd.DataFrame(features, columns=names)

    # save
    os.makedirs(os.path.dirname(savepath), exist_ok=True)
    with open(savepath, 'wb') as f:
        pickle.dump(df, f)