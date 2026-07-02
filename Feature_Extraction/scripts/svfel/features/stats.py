import numpy as np
from scipy.stats import skew, kurtosis, iqr, median_abs_deviation


def xmean(frames, axis=-1):
    # arithmetic mean
    return np.mean(frames, axis)
    

def xstd(frames, axis=-1):
    # arithmetic standard deviation
    return np.std(frames, axis)


def xmoment(frames, m, axis=-1):
    # central moments
    return xmean((frames - xmean(frames, axis))**m, axis)


def xmean_abs_dev(frames, axis=-1):
    # mean absolute deviation - 1st central absolute moment
    return xmean(np.abs(frames - np.mean(frames, axis, keepdims=True)), axis)


def xvar(frames, axis=-1):
    # variation - 2nd central moment
    return np.var(frames, axis)


def xskew(frames, axis=-1):
    # skewness - 3rd standardized moment
    # fisher's moment coefficient
    return xmean(((frames - np.mean(frames, axis, keepdims=True)) / np.std(frames, axis=-1, keepdims=True))**3, axis)
   
 
def xkurt(frames, axis=-1, fisher=True):
    # kurtosis - 4th standardized moment
    k = xmean(((frames - np.mean(frames, axis, keepdims=True)) / np.std(frames, axis=-1, keepdims=True))**4, axis)
    if fisher:
        return k - 3
    else:
        return k


def xmedian(frames, axis=-1):
    # median
    return np.median(frames, axis)


def xquantile(frames, q, axis=-1):
    # quantile q
    return np.quantile(frames, q, axis)


def xmedian_abs_dev(frames, axis=-1):
    # median absolute deviation
    return xmedian(np.abs(frames - np.median(frames, axis, keepdims=True)), axis) 


def xiqr(frames, rng=[0.25, 0.75], axis=-1):
    # interquartile range
    q1 = xquantile(frames, rng[0], axis)
    q3 = xquantile(frames, rng[1], axis)
    return q3 - q1
   
    
def xmax(frames, axis=-1):
    # maximum
    return np.max(frames, axis)


def xmin(frames, axis=-1):
    # minimum
    return np.min(frames, axis)


def xrange(frames, axis=-1):
    # value range
    return xmax(frames, axis) - xmin(frames, axis)


def aggregate_func(frames, axis=-1):
    features = [xmean(frames, axis),
                xstd(frames, axis),
                xmean_abs_dev(frames, axis),
                xskew(frames, axis),
                xkurt(frames, axis),
                xmedian(frames, axis),
                xquantile(frames, 0.25, axis),
                xquantile(frames, 0.75, axis),
                xmedian_abs_dev(frames, axis),
                xiqr(frames, [0.25, 0.75], axis),
                xmax(frames, axis),
                xmin(frames, axis),
                xrange(frames, axis)]
    names = ['mean', 
             'std', 
             'mean_abs_dev',
             'skewness',
             'kurtosis',
             'median',
             'q1',
             'q3',
             'median_abs_dev',
             'interquartile range',
             'max',
             'min',
             'range']
    return features, names
