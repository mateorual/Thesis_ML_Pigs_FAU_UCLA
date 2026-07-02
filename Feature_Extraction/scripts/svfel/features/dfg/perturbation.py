import numpy as np
import matplotlib.pyplot as plt


def _relative_tdiff(X):
    return np.mean(np.abs(np.diff(X, n=1, axis=-1)), axis=-1) / np.mean(np.abs(X), axis=-1)

def jit_percent(X):
    # jitter percent
    if X.ndim == 1:
        X = X.reshape(1, -1)
    return _relative_tdiff(X) * 100.
    
def jit_factor(X):
    # jitter factor
    if X.ndim == 1:
        X = X.reshape(1, -1)
    return _relative_tdiff(1./X) * 100.
    
def jit_ratio(X):
    # jitter ratio
    if X.ndim == 1:
        X = X.reshape(1, -1)
    return _relative_tdiff(X) * 1000.

def shim_percent(X):
    # shimmer percent
    if X.ndim == 1:
        X = X.reshape(1, -1)
    return _relative_tdiff(X) * 100.


def _perturbation_factor(X):
    return np.mean(np.abs((np.diff(X, n=1, axis=-1)) / X[:, 1:]), axis=-1)

def perturbation_factor(X):
    # amplitude / period / energy perturbation factor
    if X.ndim == 1:
        X = X.reshape(1, -1)
    return _perturbation_factor(X) * 100.


def _perturbation_quotient(X, k):
    assert k in [3,5,11]
    assert X.shape[-1] >= k
    l = int((k-1)/2)
    start = l
    end = int(X.shape[-1]-l)
    
    num = k * X[:, start:end]
    den = np.stack([np.sum(X[:, i-l:i+l+1], axis=-1) for i in range(start,end)]).T
    
    pqk = np.mean(np.abs(1. - num / den), axis=-1)
    return pqk

def perturbation_quotient(X, k):
    # amplitude / period / energy perturbation quotient
    if X.ndim == 1:
        X = X.reshape(1, -1)
    return _perturbation_quotient(X, k) * 100.


def _variability_index(X):
    vi = np.mean(np.square(X - np.mean(X, axis=-1, keepdims=True)), axis=-1) / np.square(np.mean(X, axis=-1))
    return vi


def avi(X):
    # amplitude variability index
    if X.ndim == 1:
        X = X.reshape(1, -1)
    return np.log10(_variability_index(X) * 1000.)

def pvi(X):
    # period variability index
    if X.ndim == 1:
        X = X.reshape(1, -1)
    return _variability_index(X) * 1000.


def mean_shim(X):
    # mean shimmer
    if X.ndim == 1:
        X = X.reshape(1, -1)
    mshim = 20.*np.mean(np.abs(np.log10(X[:, :-1]/X[:, 1:])), axis=-1)
    # mshim = 20.*np.mean(np.abs(np.log10(a[:-1])+np.log10(1./a[1:])))
    return mshim

def mean_jit(X):
    # mean jitter
    if X.ndim == 1:
        X = X.reshape(1, -1)
    return np.mean(np.abs(np.diff(X, n=1, axis=-1)), axis=-1)

def _rap(X, v=1):
    # relative average perturbation
    if X.ndim == 1:
        X = X.reshape(1, -1)
    rap = []
    for x in X:
        x_rap = np.sum([np.abs(np.mean(x[i-1:i+2]) - x[i]) for i in range(1,len(x)-1)]) / np.sum(x)
        if v == 2:
            x_rap = x_rap * len(x) / (len(x)-2.)
        rap.append(x_rap)
    return np.stack(rap)






### ALLE PARAMETER WIE GAT (WENN ALTER PERTRUBATION-QUOTIENT BENUTZT WIRD)
### STD VON CYCLE DURATION ANDERS, OBWOHL CYCLE DURATIONS GLEICH -> Berechnungsfehler std in gat?


