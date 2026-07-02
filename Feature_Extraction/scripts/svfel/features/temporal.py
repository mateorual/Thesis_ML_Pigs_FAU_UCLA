from svfel.utils import util

import numpy as np

# https://github.com/fraunhoferportugal/tsfel/blob/21b023aae69fa1a1ff9abe3c534980fa7883025f/tsfel/feature_extraction/features.py


def tdiff(frames, n):
    # temporal difference
    return np.diff(frames, n, axis=-1)


def p2p_amplitude(frames):
    # peak to peak amplitude
    return np.abs(np.max(frames, axis=-1, keepdims=True) - np.min(frames, axis=-1, keepdims=True))


def total_energy(frames, fs):
    # signal energy
    e = np.sum(np.square(frames), axis=-1, keepdims=True) / (frames.shape[-1]/fs)
    return e


def zcr(frames):
    # zero crossing rate
    rate = [len(np.where(np.diff(np.sign(frame)))[0]) / len(frame) for frame in frames]
    return np.stack(rate).reshape(-1, 1)


def temporal_entropy(frames, prob='kde'):
    # shannon entropy
    # https://github.com/fraunhoferportugal/tsfel/blob/21b023aae69fa1a1ff9abe3c534980fa7883025f/tsfel/
    
    if prob == 'kde':
        p = [util.kde(frame) for frame in frames]
    elif prob == 'gauss':
        p = [util.gaussian(frame) for frame in frames]
    p = np.stack(p)
    return - np.sum(util.xlogx(p), axis=-1, keepdims=True) / np.log2(frames.shape[-1])


def cp2p_amplitude(cframes):
    # peak to peak amplitude
    p2p = np.stack([np.abs(np.max(cframe, keepdims=True) - np.min(cframe, keepdims=True)) for cframe in cframes])
    return p2p


def ctotal_energy(cframes, fs):
    # signal energy
    e = np.stack([np.sum(np.square(cframe), keepdims=True) / (len(cframe)/fs) for cframe in cframes])
    return e

def cpitch(cframes, fs):
    t = np.stack([float(len(cframe))/fs for cframe in cframes]).reshape(-1,1)
    f = 1./t
    return t, f

def ctemporal_entropy(cframes, prob='kde'):
    # shannon entropy
    # https://github.com/fraunhoferportugal/tsfel/blob/21b023aae69fa1a1ff9abe3c534980fa7883025f/tsfel/
    
    if prob == 'kde':
        p = [util.kde(cframe) for cframe in cframes]
    elif prob == 'gauss':
        p = [util.gaussian(cframe) for cframe in cframes]
    
    e = []
    for p_ in p:
        e.append(-np.sum(util.xlogx(p_), axis=-1, keepdims=True) / np.log2(p_.shape[-1]))
    
    return np.stack(e)

##### machen keinen wirklichen sinn, unterschiede extrem gering

# def temporal_centroid(frames, fs):
#     # https://github.com/fraunhoferportugal/tsfel/blob/21b023aae69fa1a1ff9abe3c534980fa7883025f/tsfel/
    
#     t = np.array([float(x)/fs for x in range(frames.shape[-1])], dtype=np.float64)
    
#     energy = frames ** 2
#     t_energy = np.dot(energy, t) #dim?
#     energy_sum = np.sum(energy, axis=-1)
#     centroid = t_energy / energy_sum
#     return centroid


# def temporal_slope(frames):
#     # https://github.com/fraunhoferportugal/tsfel/blob/21b023aae69fa1a1ff9abe3c534980fa7883025f/tsfel/   
#     # linear trend of the signal
#     x = np.linspace(0, frames.shape[-1] - 1, frames.shape[-1])
#     return np.polyfit(x, frames.T, deg=1)[0]


# def rms(frames):
#     # root mean square
#     return np.sqrt(np.mean(np.square(frames), axis=-1))
