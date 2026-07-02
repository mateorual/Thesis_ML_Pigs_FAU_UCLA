from svfel.utils import tft, util

import numpy as np
from scipy.linalg import toeplitz, inv
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

def lpc(frames, order):
    '''
    Linear Prediction Coefficients

    Parameters
    ----------
    frames : array [n_frames, n_features]
        (windowed) input signal
    order : int
        lpc order

    Returns
    -------
    a : array [n_frames, n_features]
        order + 1 linear prediction coefficients
    e : array [n_frames, 1]
        prediction error
    
    Sources:
    https://github.com/cournape/talkbox/blob/master/scikits/talkbox/linpred/py_lpc.py
    https://github.com/RicherMans/pymir/blob/master/pymir/LinearPredictiveAnalysis.py
    '''
    a, e = [], []
    for frame in frames:
        # autocorrelation 
        r = tft.acorr(frame, order)  
        # linear prediction coefficients, error
        ai, ei = _lpc(r, order)       
        # append
        a.append(ai)
        e.append(ei)        
    # concat
    a = np.stack(a)
    e = np.array(e)  # Keep as 1D array so ei is scalar when iterating
    return a, e


def _lpc(r, order):
    # lpc from autocorrelation coefficients 
    
    # relevant lags
    if r.shape[-1] > order:
        r = r[:order+1]
    
    # linear prediction coefficients from inv toeplitz matrix
    # a = np.dot(np.linalg.pinv(toeplitz(acseq[:-1])), -acseq[1:].T)
    a = np.dot(inv(toeplitz(r[:-1])), -r[1:])

    # squared prediction error (not exactly the same as sum(s-s_hat)**2 but close)
    e = r[0] + np.dot(r[1:], a)
    
    # first coefficient is always 1
    a = np.hstack([1., a])
    return a, e


def lpcc(a, e):
    # linear prediction cepstral coefficients
    # a = lpc coefs, e = lpc squared error
    
    # remove constant coefficient
    assert all(a[:, 0] == 1)
    a = a[:, 1:]
    
    cc = []
    for ai, ei in zip(a, e):
        # linear prediction cepstral coefficients
        cc.append(_lpcc(ai, ei))
    # concat
    cc = np.concatenate(cc)
    return cc
    
    
def _lpcc(a, e):
    # linear prediction cepstral coefficients
    # a = lpc coefs, e = lpc squared error
    # without const. 1   

    # get order 
    order = int(a.shape[-1])
    
    # first two cepstral coefficients
    cc = [np.log(e), # c_0 = ln(p)
          -a[0]]     # c_1 = -a_1
    
    # other coefficients: c_i = -a_i - sum_i^n-1(i/n * a_n-i * c_i)
    for n in range(2, order+1):
        # use order + 1 as upper bound for the last iteration
        upbound = (order + 1 if n > order else n)       
        c = -np.sum(i * cc[i] * a[n - i - 1] for i in range(1, upbound)) * 1. / upbound        
        c -= a[n - 1] if n <= order else 0.        
        cc.append(c)
    
    cc = np.array(cc, dtype=a.dtype)
    
    return cc[np.newaxis, ...]

def lsp(A):
    # https://github.com/RicherMans/pymir/blob/master/pymir/LinearPredictiveAnalysis.py
    # https://pyspectrum.readthedocs.io/en/latest/_modules/spectrum/linear_prediction.html
    # convert linear prediction coefficients to line spectral pairs / line spectral frequencies
    # a[0] == 1!
    # p_m = a_m + a_M+1-m (vocal tract with glottis closed)
    # q_m = a_m - a_M+1-m (vocal tract with glottis open)

    lsf = []
    for ai in A:
        # convert to line spectral pairs / frequencies
        lsf.append(_lsp(ai))
    # concat
    lsf = np.concatenate(lsf)

    return lsf
    
def _lsp(a):
    # https://github.com/RicherMans/pymir/blob/master/pymir/LinearPredictiveAnalysis.py
    # https://pyspectrum.readthedocs.io/en/latest/_modules/spectrum/linear_prediction.html
    # convert linear prediction coefficients to line spectral pairs / line spectral frequencies    
    # a[0] == 1!
    # p_m = a_m + a_M+1-m (vocal tract with glottis closed)
    # q_m = a_m - a_M+1-m (vocal tract with glottis open)
    a1 = np.hstack((a, np.zeros(1, dtype=a.dtype)))
    a2 = a1[::-1]    
    P = a1 + a2
    Q = a1 - a2
    # get roots
    roots_p = np.roots(P[::-1])
    roots_q = np.roots(Q[::-1])
    # convert to angles
    lsf_p = np.angle(roots_p)
    lsf_q = np.angle(roots_q)    
    # only keep positive elements
    lsf_p = lsf_p[lsf_p > 0]
    lsf_q = lsf_q[lsf_q > 0]
    # sort, remove last element (always pi)
    lsf = np.sort(np.hstack((lsf_p, lsf_q)))[:-1]
    
    return lsf[np.newaxis, ...]


def lpc_to_formant(A, fs):
    
    formants = []
    
    for ai in A:
        
        formants.append(_lpc_to_formant(ai, fs))
    
        
    #formants = np.concatenate(formants)
    
    return formants


def _lpc_to_formant(a, fs):
    # https://de.mathworks.com/help/signal/ug/formant-estimation-with-lpc-coefficients.html
    # https://www.fon.hum.uva.nl/praat/manual/Source-filter_synthesis.html
    
    # get roots of a(z)=0, only keep roots with one sign for imaginary part
    roots = np.roots(a)
    roots = roots[np.imag(roots)>=0]    
    # get corresponding angles
    angles = np.angle(roots)
    # convert to hz
    freqs = util.rad_to_hz(angles, fs)
    # sort
    freqs = sorted(freqs)
    index = np.argsort(freqs)
    # get bandwidths of formants (represented by distance of prediction polynomial zeros from unit circle)
    bandwidths = -0.5 * fs / (2*np.pi) * np.log(np.abs(roots[index]))
    
    # apply criteria
    # matlab: f>90, bw<400
    # praat: f>50, f<fs-50
    # formants = np.array([f for (f, bw) in zip(freqs, bandwidths) if (f > 50 and f < fs-50 and bw < 400)], dtype=a.dtype)
    formants = np.array([f for (f, bw) in zip(freqs, bandwidths) if (f > 50. and f < fs/2-50.)], dtype=a.dtype)
    
    # return formants
    return formants#[np.newaxis, ...]

def get_order(n_formants):
    return int(2 + n_formants*2)

def get_formants(frames, fs, n_formants=3):

    # pre-emphasis (twice??)
    frames_pe = util.pre_emphasize(util.pre_emphasize(frames))
    # frames_pe = util.pre_emphasize(frames)
    # frames_pe = frames
    # lpc
    # number of lpc coefficients ???
    # n_coef = int(2+fs/1000) # internet
    n_coef = int(2 + n_formants*2) # lecture/matlab

    a, e = lpc(frames_pe, order=n_coef)

    # get formants
    formants = lpc_to_formant(a, fs)
    formants = [form[:n_formants] for form in formants]

    return formants, a, e


def get_f2_f3_a(frames, fs, n_formants=5, debug=False):
    
    ### !!! this is fine-tuned to vowel /a/ ("Pfusch" - Hauptwort)
    
    # pre-emphasis
    frames_pe = util.pre_emphasize(util.pre_emphasize(frames))

    # number of lpc coefficients
    # n_coef = int(2+fs/1000) # internet
    n_coef = int(2 + n_formants*2) # lecture/matlab
    # lpc
    a, e = lpc(frames_pe, order=n_coef)
    # spectral envelope
    envs, bins = lpc_response(a, e, fs)
    # alternative formants
    formants_alt = lpc_to_formant(a, fs)
    
    ### find F2 / F3
    formants = []
    # for each lpc envelope
    for i, env in enumerate(envs):
        # find peaks = formants
        peaks = find_peaks(env)[0]
        # at least 3 formants needed
        if len(peaks) > 2:
            # at least 2 formants before 2000Hz (vowel a)
            if bins[peaks[1]] < 2000:
                # append 2nd & 3rd formant frequency
                formants.append(bins[peaks[1:3]])
                if debug:
                    plt.figure(figsize=(10,5))
                    plt.plot(bins, env, label='lpc')
                    plt.scatter(bins[peaks], env[peaks], c='k', zorder=3)
                    plt.scatter(bins[peaks[1]], env[peaks[1]], c='r', label='f2', zorder=3)
                    plt.scatter(bins[peaks[2]], env[peaks[2]], c='g', label='f3', zorder=3)
                    plt.xlim([0,5600])
                    plt.show()
            else:
                # use alternative 2nd formant (usually correct)
                f2 = formants_alt[i][1]
                # use 2nd peak as 3rd formant
                f3 = bins[peaks[1]]
                formants.append(np.array([f2,f3]))
                # !!! append 1st & 2nd formant frequency ???
                # if bins[peaks[0]] > 950:
                # formants.append(bins[peaks[:2]])
                # else:
                #     formants.append(None)
                
                if debug:
                    plt.figure(figsize=(10,5))
                    plt.plot(bins, env, label='lpc')
                    plt.scatter(bins[peaks], env[peaks], c='k', zorder=3)
                    plt.scatter(bins[peaks[0]], env[peaks[0]], c='r', label='f2', zorder=3)
                    plt.scatter(bins[peaks[1]], env[peaks[1]], c='g', label='f3', zorder=3)
                    plt.xlim([0,5600])
                    plt.show()
        else:
            formants.append(None)

    return formants, a, e


def lpc_response(A, E, fs, norm=False):
    
    envs = []
    for ai, ei in zip(A,E):
        w, h = util.digital_filter_response(ei,ai)
        env = np.abs(h)
        if norm: env /= np.sum(env, axis=-1)
        envs.append(env)
        # bins.append(utils.rad_to_hz(w, fs))
        
    envs = np.stack(envs)
    bins = util.rad_to_hz(w, fs)
    
    return envs, bins


### LEVINSON
def lpc_lev(signal, order):
    # https://github.com/cournape/talkbox/blob/master/scikits/talkbox/linpred/levinson_lpc.py
    """Compute the Linear Prediction Coefficients.
    Return the order + 1 LPC coefficients for the signal. c = lpc(x, k) will
    find the k+1 coefficients of a k order linear filter:
      xp[n] = -c[1] * x[n-2] - ... - c[k-1] * x[n-k-1]
    Such as the sum of the squared-error e[i] = xp[i] - x[i] is minimized.
    Parameters
    ----------
    signal: array_like
        input signal
    order : int
        LPC order (the output will have order + 1 items)
    Returns
    -------
    a : array-like
        the solution of the inversion.
    e : array-like
        the prediction error.
    k : array-like
        reflection coefficients.
    Notes
    -----
    This uses Levinson-Durbin recursion for the autocorrelation matrix
    inversion, and fft for the autocorrelation computation.
    For small order, particularly if order << signal size, direct computation
    of the autocorrelation is faster: use levinson and correlate in this case."""
    n = signal.shape[0]
    if order > n:
        raise ValueError("Input signal must have length >= order")

    r = tft.acorr_fft(signal, order)
    return levinson_1d(r, order)


def levinson_1d(r, order):
    # https://github.com/cournape/talkbox/blob/master/scikits/talkbox/linpred/py_lpc.py
    """Levinson-Durbin recursion, to efficiently solve symmetric linear systems
    with toeplitz structure.
    Parameters
    ---------
    r : array-like
        input array to invert (since the matrix is symmetric Toeplitz, the
        corresponding pxp matrix is defined by p items only). Generally the
        autocorrelation of the signal for linear prediction coefficients
        estimation. The first item must be a non zero real.
    Notes
    ----
    This implementation is in python, hence unsuitable for any serious
    computation. Use it as educational and reference purpose only.
    Levinson is a well-known algorithm to solve the Hermitian toeplitz
    equation:
                       _          _
        -R[1] = R[0]   R[1]   ... R[p-1]    a[1]
         :      :      :          :      *  :
         :      :      :          _      *  :
        -R[p] = R[p-1] R[p-2] ... R[0]      a[p]
                       _
    with respect to a (  is the complex conjugate). Using the special symmetry
    in the matrix, the inversion can be done in O(p^2) instead of O(p^3).
    """
    r = np.atleast_1d(r)
    if r.ndim > 1:
        raise ValueError("Only rank 1 are supported for now.")

    n = r.size
    if n < 1:
        raise ValueError("Cannot operate on empty array !")
    elif order > n - 1:
        raise ValueError("Order should be <= size-1")

    if not np.isreal(r[0]):
        raise ValueError("First item of input must be real.")
    elif not np.isfinite(1/r[0]):
        raise ValueError("First item should be != 0")

    # Estimated coefficients
    a = np.empty(order+1, r.dtype)
    # temporary array
    t = np.empty(order+1, r.dtype)
    # Reflection coefficients
    k = np.empty(order, r.dtype)

    a[0] = 1.
    e = r[0]

    for i in range(1, order+1):
        acc = r[i]
        for j in range(1, i):
            acc += a[j] * r[i-j]
        k[i-1] = -acc / e
        a[i] = k[i-1]

        for j in range(order):
            t[j] = a[j]

        for j in range(1, i):
            a[j] += k[i-1] * np.conj(t[i-j])

        e *= 1 - k[i-1] * np.conj(k[i-1])

    return a, e, k

