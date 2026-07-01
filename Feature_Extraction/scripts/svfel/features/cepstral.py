from svfel.utils import fbanks, util, tft
from svfel.features import lpc

import numpy as np
from scipy.signal import lfilter
from scipy.sparse import csr_matrix

###############################################################################

# MFCC

###############################################################################
def mfcc(frames, fs, nfft, nmfcc, nfilt=128, lifter=0, fmin=0.0, fmax=None, s_norm=False, debug=False):
    # spectrogram
    S, xf = tft.rspectrogram(frames, fs, nfft, power=1., norm=s_norm)
    # filterbank (librosa uses different mel conversion for non-htk)
    fbank, fc = fbanks.mel_filterbank(fs, nfft, nfilt, fmin, fmax, debug=False)
    # apply mel filterbank
    S_mel = fbanks.apply(S, fbank)
    S_mel = util.amp_to_db(S_mel)
    # dct
    mfcc = tft._dct(S_mel)[:, 1:nmfcc+1] # first/second feature different from librosa
    # liftering (not really necessary?)
    # https://dsp.stackexchange.com/questions/26019/sinusoidal-liftering-in-implementations-of-mfcc
    if lifter > 0:
        mfcc *= lifter_htk(lifter, nmfcc)
        # plt.plot(utils.lifter_htk(lifter, nmfcc))
        # plt.show()

    # normalization??? no
    # -> either lifter or norm, because lifter is const. factor
    # mfcc = utils.zscore_norm(mfcc, axis=0)
    # mfcc = utils.mean_center(mfcc, axis=0)
    # mfcc = utils.minmax_norm(mfcc, axis=0) # only one that would make sense (exept for min/max)
    
    if debug:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8,8))
        plt.subplot(311)
        for f in fbank:
            plt.plot(xf, f)
        plt.title('FILTERBANK')
        plt.subplot(312)
        plt.imshow(S_mel.T, interpolation='nearest', aspect='auto', origin='lower', 
                   extent=[0, len(S_mel), 0, fc.max()])
        ax = plt.gca()
        ax.axes.set_yticks(fc[::4])
        plt.title('MEL-SPECTROGRAM')
        plt.subplot(313)
        plt.imshow(mfcc.T, interpolation='nearest', aspect='auto', origin='lower')
        plt.title('MFCC')
        plt.show()
        # plt.plot(lifter_htk(lifter, nmfcc))
        # plt.show()
        
    return mfcc, S_mel

###############################################################################

# GFCC

###############################################################################
def gfcc(frames, fs, nfft, ngfcc=12, nfilt=64, lifter=0, fmin=0.0, fmax=None, s_norm=False, debug=False):
    # spectrogram
    S, xf = tft.rspectrogram(frames, fs, nfft, power=1., norm=s_norm)
    # gammatone filterbank
    fbank, _ = fbanks.gammatone_filterbank(fs, nfft, nfilt, fmin, fmax, width=1., erbtype='glasberg', debug=debug)
    # gammatone spectrogram
    S_gamma = fbanks.apply(S, fbank)
    # to db
    S_gamma = util.amp_to_db(S_gamma)
    # dct
    # gfcc = utils._dct(S_gamma)[:, 1:ngfcc+1]
    cc = tft._dct(S_gamma)[:, :ngfcc]
    
    # lifter
    if lifter > 0:
        cc *= lifter_htk(lifter, ngfcc)
    
    if debug:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12,8))
        plt.subplot(121)
        plt.imshow(S_gamma.T, interpolation='nearest', aspect='auto', origin='lower')
        plt.colorbar()
        plt.title('Gammatone Log Spectrogram')
        plt.subplot(122)
        plt.imshow(cc.T, interpolation='nearest', aspect='auto', origin='lower')
        plt.colorbar()
        plt.title('GFCC')
        plt.show()
        # plot fb-spec
    
    return cc, S_gamma

###############################################################################

# PLP

###############################################################################
def plpcc(a, e, lifter=0):
    ### cepstral coefficients
    # Finally, (viii), cepstral coefficients are obtained from
    # the predictor coefficients by a recursion that is equivalent to the
    # logarithm of the model spectrum followed by an inverse Fourier
    # transform. 
    cc = lpc.lpcc(a,e)
    # liftering (not really necessary?)
    # https://dsp.stackexchange.com/questions/26019/sinusoidal-liftering-in-implementations-of-mfcc
    if lifter > 0:
        cc *= lifter_htk(lifter, cc.shape[-1])
    return cc


def plp(frames, fs, nfft=None, lpc_order=12, nfilts=None, use_rasta=False, width=1., fmin=0.0, fmax=None, debug=False):
    
    ### power spectrum computation + critical band analysis
    # (i) The power spectrum is computed from the windowed speech signal.
    # (ii) A frequency warping into the Bark scale is applied.
    # (iii) The auditorily warped spectrum is convoluted with the power spectrum
    # of the simulated critical-band masking curve to simulate the
    # critical-band integration of human hearing. 
    # (iv) The smoothed spectrum is down-sampled at intervals of ≈ 1 Bark. 
    # The three steps frequency warping, smoothing and sampling (ii-iv) are 
    # integrated into a single filter-bank called Bark filter-bank. 
    S, xf = tft.rspectrogram(frames, fs, nfft, power=2.)
    fbank, _ = fbanks.bark_filterbank(fs, nfft, nfilts, width, fmin, fmax, debug=debug)
    S_plp = fbanks.apply(S, fbank)

    ## RASTA
    if use_rasta:
        # nonlinear compression
        S_plp = np.log(S_plp)

        # RASTA filtering
        S_plp = rasta(S_plp)

        # nonlinear expansion
        S_plp = np.exp(S_plp)
    
    ### equal loudness preemphasis
    # (v) An equal-loudness pre-emphasis weights the filter-bank outputs 
    # to simulate the sensitivity of hearing.  
    S_plp = equal_loudness_preemphasis(S_plp, fs, fmin, fmax, debug)
    
    ### intensity loudness conversion
    # (vi) The equalized values are transformed according to the power law 
    # of Stevens by raising each to the power of 0.33. 
    S_plp = intensity_loudness_conversion(S_plp)
       
    ### autocorrelation
    nbands = S_plp.shape[-1]
    r = np.real(np.fft.ifft(np.hstack((S_plp, S_plp[:,np.arange(nbands-2,0,-1)]))))    
    r = r[:, :nbands]
    
    ### linear prediction 
    # The resulting auditorily warped line
    # spectrum is further processed by (vii) linear prediction (LP).
    # Precisely speaking, applying LP to the auditorily warped line
    # spectrum means that we compute the predictor coefficients of a
    # (hypothetical) signal that has this warped spectrum as a power
    # spectrum. 
    a, e = [], []
    for ri in r:
        # lpc
        ai, ei = lpc._lpc(ri, order=lpc_order)
        a.append(ai)
        e.append(ei)        
    a = np.stack(a) # linear prediction coefficients
    e = np.array(e) # error - keep as 1D array so ei is scalar when iterating

    return a, e, S_plp


def equal_loudness_preemphasis(spectrum, fs, fmin=0.0, fmax=None, debug=False):    
    # max freq. (default = nyquist)
    if fmax == None:
        fmax = float(fs/2.)    
    
    # number of frames, number of bark bands
    nframes, nbands = spectrum.shape
    
    # band center frequencies
    fc = util.bark_to_hz(np.linspace(util.hz_to_bark(fmin), util.hz_to_bark(fmax), num=nbands))
    
    # Hynek's magic equal-loudness-curve formula   
    w = (2.*np.pi*fc)**2
    eql = ((w + 56.8e6) * w**2) / ((w + 6.3e6)**2 * (w + 0.38e9))
    # !!!
    # should be used if nyq>5000 but has weird range
    # eql1 = ((w + 56.8e6) * w**2) / ((w + 6.3e6)**2 * (w + 0.38e9) * (w**3 + 9.58e26)) 

    # weight the critical bands
    spectrum_eql = np.tile(eql.T,(nframes,1)) * spectrum    
    # replace first & last band with nearest neighbor
    spectrum_eql = spectrum_eql[:, np.hstack((1,np.arange(1, nbands - 1), nbands - 2))]
    
    if debug:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(fc, eql)
        plt.title('Equal Loudness Curve')
        plt.show()
    
    return spectrum_eql
    
    
def intensity_loudness_conversion(spectrum):
    return spectrum**.33
    

def rasta(x):
    """Apply RASTA filtering to the input signal.
    
    :param x: the input audio signal to filter.
        cols of x = critical bands, rows of x = frame
        same for y but after filtering
        default filter is single pole at 0.94
    """
    x = x.T
    numerator = np.arange(.2, -.3, -.1)
    denominator = np.array([1, -0.94])

    # Initialize the state.  This avoids a big spike at the beginning
    # resulting from the dc offset level in each band.
    # (this is effectively what rasta/rasta_filt.c does).
    # Because Matlab uses a DF2Trans implementation, we have to
    # specify the FIR part to get the state right (but not the IIR part)
    y = np.zeros(x.shape)
    zf = np.zeros((x.shape[0], 4))
    for i in range(y.shape[0]):
        y[i, :4], zf[i, :4] = lfilter(numerator, 1, x[i, :4], axis=-1, zi=[0, 0, 0, 0])

    # .. but don't keep any of these values, just output zero at the beginning
    y = np.zeros(x.shape)

    # Apply the full filter to the rest of the signal, append it
    for i in range(y.shape[0]):
        y[i, 4:] = lfilter(numerator, denominator, x[i, 4:], axis=-1, zi=zf[i, :])[0]
    
    return y.T


###############################################################################

# PNCC

###############################################################################

def pncc(frames, fs, nfft, nceps=12, nfilt=40, simple=False, fmin=0.0, fmax=None, debug=False):
    
    # pre-emphasis
    frames = util.pre_emphasize(frames, a=0.97)
    
    # spectrogram (power 2)
    S, xf = tft.rspectrogram(frames, fs, nfft, power=2.)
    
    # filterbank 
    fbank, _ = fbanks.gammatone_filterbank(fs, nfft, nfilt, fmin, fmax, width=1., erbtype='glasberg', debug=debug)
    
    # gammatone frequency integration
    P_gt = fbanks.apply(S, fbank)
    
    if not simple:        
        # medium-time processing
        S, R = medium_time_processing(P_gt, nfilt)
        # apply time- & frequency averaged transfer function
        T = P_gt * S
    else:
        T = P_gt
    
    # mean power normalization
    U = mean_power_normalization(T, L=nfilt)
    
    # power function nonlinearity
    V = U**(1./15.)
    
    # dct
    pncc = tft._dct(V)[:, :nceps]
    
    # liftering
    # ?
    
    # mean normalization
    # ?
    
    return pncc, V


def _medium_time_power(P, m=2):
    # zero padding
    P_pad = np.pad(P, [(m, m), (0, 0)], 'constant')
    # moving average: 2*m+1 time frames
    Q = []
    for i in range(m, len(P)+m):
        Q.append(1. / (2. * float(m) + 1.) * np.sum(P_pad[i-m:i+m+1, :], axis=0))
    return np.stack(Q)


def _asymmetric_lowpass(Q_in, lambda_a=0.999, lambda_b=0.5):
    Q_le = [0.9 * Q_in[0,:]]   
    for q in Q_in[1:]:
        in_gt_out = lambda_a * Q_le[-1] + (1 - lambda_a) * q
        in_lt_out = lambda_b * Q_le[-1] + (1 - lambda_b) * q
        Q_le.append(np.where(q >= Q_le[-1], in_gt_out, in_lt_out))
    return np.stack(Q_le)


def _temporal_masking(Q_in, lambda_t=0.85, mu_t=0.2):
    # temporal masked signal, online peak power
    R_sp, Q_p = [Q_in[0, :]], [Q_in[0, :]]
    for q in Q_in[1:]:        
        R_sp.append(np.where(q >= lambda_t * Q_p[-1], q, mu_t * Q_p[-1]))
        Q_p.append(np.maximum(lambda_t * Q_p[-1], q))   
    return np.stack(R_sp)
    

def _weight_smoothing(R, Q, L=40, N=4):
    S = []
    for r, q in zip(R,Q):
        s = []
        for l in range(len(r)):
            l1 = max(l - N, 0)
            l2 = min(l + N, L)
            s.append(1. / float(l2 - l1 + 1.) * np.sum(r[l1:l2] / q[l1:l2]))
        S.append(np.stack(s))
    S = np.stack(S)
    return S


def medium_time_processing(P, nfilt):
    ### medium time power calculation
    Q = _medium_time_power(P)
    
    ### asymmetric noise suppression with temporal masking
    # lower envelope
    Q_le = _asymmetric_lowpass(Q, lambda_a=0.999, lambda_b=0.5)
    # subtract
    Q_sub = Q - Q_le
    # halfwave rectification
    Q_0 = np.where(Q_sub < 0., 0., Q_sub)
    # floor level
    Q_f = _asymmetric_lowpass(Q_0, lambda_a=0.999, lambda_b=0.5)
    # temporal masking
    R_sp = _temporal_masking(Q_0, lambda_t=0.85, mu_t=0.2)
    # excitation (Q >= cQle -> R=Rsp) vs non-excitation (Q < cQle -> R=Qf)
    c = 2.
    R = np.where(Q >= c * Q_le, R_sp, Q_f)
    
    ### weight smoothing 
    S = _weight_smoothing(R, Q, L=nfilt, N=4)
    
    # for a,b,c,d in zip(P, Q, R, S):
    #     plt.figure()
    #     plt.plot(a, label='P')
    #     plt.plot(b, label='Q')
    #     plt.plot(c, label='R')
    #     plt.plot(d, label='S')
    #     plt.legend()
    #     plt.show()
        
    return S, R


def mean_power_normalization(T, L=40, lambda_mu=0.999, k=1):
    # initial value = "value obtained from the training database"
    mu = [np.mean(T[0])] #?
    
    for t in T[1:]:
        # c*mu_old + (1-c)/L*sum_new
        mu.append(lambda_mu * mu[-1] + (1 - lambda_mu) * np.mean(t))
    
    U = k * T / np.stack(mu).reshape(-1,1)
    
    # U = T / np.mean(T) #?
    
    return U


###############################################################################

# DELTA CC

###############################################################################
def delta_cc(cc, n=1):
    # nframes, nfeats
    return np.diff(cc, n, axis=-2)

###############################################################################

# LIFTER

###############################################################################
def lifter_htk(l, n, dtype=np.float64):
    # l - lifter coef
    # n - ncepstra
    return  1 + (l / 2) * np.sin(np.pi * np.arange(1, n+1, dtype=dtype) / l)  


# ADDED NEW FEATURE HELPER (CQT for CQCC computation)
# reference https://www.ee.columbia.edu/~dpwe/papers/Brown91-cqt.pdf
# github https://github.com/SuperKogito/spafe/tree/master

def constant_q_transform(frames, fs, nfft=2048, fmin=0, fmax=None, n_octave=7, n_bins_per_octave=24, spec_threshold=0.005, f0=120, q_rate=1.0):
    if fmax is None:
        fmax = fs/2.0

    freqs = np.array([f0 *2 ** ((m * n_octave + n)/n_bins_per_octave) for m in range(n_octave) for n in range(n_bins_per_octave)])

    cqt_freqs = freqs[(freqs >= fmin) & (freqs <= fmax)]
    if cqt_freqs.size == 0:
        raise ValueError("No CQT bins in the requsted range")

    # Q factor
    Q = q_rate / (2 ** (1.0/n_bins_per_octave) - 1.0)

    win_len = np.ceil(Q * fs / cqt_freqs).astype(np.int64)
    win_len = win_len[win_len <= nfft]

    cqt_freqs = cqt_freqs[-len(win_len):]
    n_pitch = len(cqt_freqs)
    n_frames = frames.shape[0]

    a = np.zeros((n_pitch, nfft), dtype=np.complex128)
    kernel = np.zeros(a.shape, dtype=np.complex128)
    for k in range(n_pitch):
        Nk = win_len[k]
        fk = cqt_freqs[k]

        start = int((nfft - Nk) / 2)
        end = start + Nk

        temp = np.exp(2j * np.pi * (fk/fs) * np.arange(Nk))
        a[k, start:end] = (1/Nk) * np.hanning(Nk) * temp
        kernel[k] = np.fft.fft(a[k], nfft)

    kernel[np.abs(kernel) <= spec_threshold] = 0.0
    kernel_sparse = csr_matrix(kernel).conjugate() / nfft

    spec = np.zeros([n_frames, n_pitch], dtype=np.complex128)
    for k, frame in enumerate(frames):
        x = (
            np.r_[frame, np.zeros(nfft - len(frame))]
            if len(frame) < nfft
            else frame[0:len(frame)]
        )
        spec[k] = np.fft.fft(x, nfft) * kernel_sparse.T

    return spec

def cqcc(frames, fs, nfft=2048, ncqcc=12, lifter=0, fmin=0.0, fmax=None, s_norm=False, n_octave=7, n_bins_per_octave=24, resample_ratio=1.0, spec_threshold=0.005, f0=120, q_rate=1.0, debug=False):
    if fmax is None:
        fmax = fs/2.0

    cqt_spec = constant_q_transform(frames, fs, nfft, fmin, fmax, n_octave, n_bins_per_octave, spec_threshold, f0, q_rate)

    S_cqt = np.abs(cqt_spec)
    S_cqt_log = util.amp_to_db(S_cqt)

    if resample_ratio != 1.0:
        S_cqt_log = resample(S_cqt_log, int(S_cqt_log.shape[1] * resample_ratio), axis=1)

    if s_norm:
        S_cqt_log = S_cqt_log / (np.sum(S_cqt_log, axis=1, keepdims=True) + 1e-12)

    cqcc = tft._dct(S_cqt_log)[:, :ncqcc]

    if lifter > 0:
        cqcc *= lifter_htk(lifter, ncqcc)

    if debug:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.imshow(S_cqt_log.T, aspect="auto", origin="lower", interpolation="nearest")
        plt.title("Log CQT Spectrogram")
        plt.subplot(1, 2, 2)
        plt.imshow(cqcc.T, aspect="auto", origin="lower", interpolation="nearest")
        plt.title("CQCC")
        plt.tight_layout()
        plt.show()

    return cqcc, S_cqt_log