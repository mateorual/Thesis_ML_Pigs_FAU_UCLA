from svfel.features import fit
from svfel.utils import util, tft

import numpy as np

###############################################################################

# CPP / CPM / CPPS

###############################################################################


def cepstral_peak_x(frames, fs, f0=None, nfft=None, n_smooth_t=7, n_smooth_q=3, debug=False):
    
    # cepstral peak prominence: f0 defined, smooth=0
    # smoothed cepstral peak prominence: f0 defined, smooth>0
    # cepstral peak magnitude: f0 undefined, smooth>=0
    
    # get cepstrum
    C = tft.cepstrogram(frames, fs, nfft)
    # C = np.mean(C, axis=0, keepdims=True)
    
    # keep positive frequencies
    if not nfft: nfft = frames.shape[-1]
    xf = np.fft.rfftfreq(nfft, d=1./fs)
    C = C[:, :len(xf)]
    
    # smoothing
    # time
    if n_smooth_t > 0:
        for i in range(C.shape[0]):
            C[i] = np.mean(C[max(0, i-int(n_smooth_t/2)):min(len(C), i+int(n_smooth_t/2)+1)], axis=0)

    # quefrency
    if n_smooth_q > 0:
        for i in range(C.shape[0]):
            for j in range(C.shape[-1]):
                C[i, j] = np.mean(C[i, max(0, j-int(n_smooth_q/2)):min(C.shape[-1], j+int(n_smooth_q/2)+1)])
    
    # get peak indices
    if f0 is None:
        # cepstral peak magnitude (f0 agnostic)
        qidx = _cpm_idx(C, fs)
    else:
        # assert same length
        assert len(f0) == C.shape[0]
        # cepstral peak prominence
        qidx = _cpp_idx(C, xf, f0, fs)
    
    # linear regression 
    reg_start = round(fs*.0033) # start after 3.3ms    
    Y = C[:, reg_start:]
    x = np.arange(Y.shape[-1]) + reg_start 
    coef = fit.least_squares(Y, x, order=1)
    y_hat = fit.draw(x, coef).T
    
    # cpp
    cpx = []
    for i, r in enumerate(qidx):
        cpx.append(C[i,r] - y_hat[i,r-reg_start])
        
    # corresponding frequencies
    fbins = _q2f(xf, nfft, qidx)
    
    if debug:
        import matplotlib.pyplot as plt
        # plot
        for i, c in enumerate(C):
            plt.figure()
            plt.plot(c)
            plt.scatter(qidx[i], c[qidx[i]], c='r')
            plt.plot(x, y_hat[i])
            plt.scatter(x[qidx[i]-reg_start], y_hat[i,qidx[i]-reg_start], c='r')
            plt.show()
        
    return np.stack(cpx).reshape(-1,1), fbins
    

def _cpp_idx(C, xf, f0, fs, tol=0.05):
    
    rmin = round(fs*.0033) # start after 3.3ms
    rmax = C.shape[-1]
    tol = 0.25 # !!!
    rtol = round(C.shape[-1]*tol)
    
    indices = []
    
    # for c, f in zip(C, f0*C.shape[0]) # some error
    for c, f in zip(C, f0):
        # get f0 bin
        fdiff = np.abs(xf - f)
        f0_idx = np.argmin(fdiff)
        # translate to r0
        rcenter = round((c.shape[-1]-1)*2/f0_idx)
        # define search range
        rstart = max(rmin, rcenter-rtol)
        rstop = min(rmax, rcenter+rtol)
        # get peak idx
        if rstart <= c.shape[-1]-1: # hotfix
            indices.append(np.argmax(c[rstart:rstop]) + rstart)
    
    return np.stack(indices)


def _cpm_idx(C, fs):
    
    rmin = round(fs*.0033) # start after 3.3ms
    indices = np.argmax(C[:, rmin:], axis=-1) + rmin
    
    return indices

def _q2f(xf, nfft, qidx):
    # get frequency bin corresponding to peak quefrency
    fidx = np.round(nfft / qidx).astype('int')
    return xf[fidx]
