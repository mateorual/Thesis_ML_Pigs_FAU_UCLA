from svfel.utils import util

import numpy as np
from librosa import yin
from scipy.fftpack import dct
from scipy.signal import hilbert 

def rms(x):
    # root mean square
    return np.sqrt(np.mean(np.square(x)))


def _ff_acorr(frame, fs, fmin=50., fmax=400., frame_len=None, debug=False):
    import matplotlib.pyplot as plt
    from scipy.signal import find_peaks
    a = acorr(frame, order=int(fs/fmin)+1)
    start = int(1./fmax*fs)
    end = int(1./fmin*fs)+1
    thres = rms(a[start:end])
    # peaks = find_peaks(a[start:end], thres, distance=int(fs/fmax))[0]
    idx = np.argmax(a[start:end]) + start
    if debug:
        plt.figure()
        plt.plot(a)
        plt.plot(np.arange(end-start)+start, a[start:end])
        plt.hlines(thres, 0, len(a))
        # plt.scatter(peaks, a[peaks])
        plt.scatter(idx, a[idx])
        plt.show()
    f0 = fs/idx
    return f0
                
def _ff_yin(frame, fs, fmin=50., fmax=400., frame_len=None):
    return np.mean(yin(frame, fmin=fmin, fmax=fmax, sr=fs, frame_length=frame_len))


def ff(frames, fs, fmin=50., fmax=400., frame_len=None, method='yin'):
    if method == 'yin':
        func = _ff_yin  
    # if method == 'fft':
    #     func = _ff_fft
    if method == 'acorr':
        func = _ff_acorr
    if frame_len == None:
        frame_len = frames.shape[-1]
    f0 = np.stack([np.mean(func(frame, fs, fmin, fmax, frame_len)) for frame in frames])
    return f0
 

def acorr(x, order=None):
    # autocorrelation with the given order
    if order is None:
        order = len(x) - 1
    return np.array([np.dot(x[:len(x)-tau], x[tau:]) for tau in range(order+1)])


def acorr_fft(x, order=None):
    maxlag = x.shape[0]
    nfft = 2 ** util.nextpow2(2 * maxlag - 1)
    if order is None:
        order = nfft//2
    # r = np.real(np.fft.ifft(np.abs(np.fft.fft(x, n=nfft) ** 2)))
    r = np.real(np.fft.ifft(np.abs(np.fft.fft(x) ** 2)))
    r = r[:maxlag+1] #/ maxlag
    return r[:order+1]


def fft(x, fs, nfft=4096):
    # FFT of signal
    # number of sample points
    if not nfft: nfft = len(x)
    # frequency magnitude
    mags = np.abs(np.fft.fft(x, nfft))
    # frequency bins
    freqs = np.fft.fftfreq(nfft, d=1./fs)
    return mags, freqs

    
def spectrogram(frames, fs, nfft=4096, power=1.):
    # spectrogram
    # number of sample points
    if not nfft: nfft = frames.shape[1]
    S = np.array([fft(frame, fs, nfft)[0] for frame in frames])**power
    freqs = np.fft.fftfreq(nfft, d=1./fs)
    return S, freqs


def rfft(x, fs, nfft=4096, norm=False):
    # FFT of real signal, return positive freqs only
    # number of sample points
    if not nfft: nfft = len(x)
    # frequency magnitude (positive f)
    mags = np.abs(np.fft.rfft(x, nfft))
    # normalize ???
    if norm: mags /= np.sum(mags)
    # frequency bins (positive f)
    freqs = np.fft.rfftfreq(nfft, d=1./fs)
    return mags, freqs


def rspectrogram(frames, fs, nfft=4096, power=1., norm=False):
    # REAL spectrogram
    # number of sample points
    if not nfft: nfft = frames.shape[1]
    S = np.array([rfft(frame, fs, nfft, norm)[0] for frame in frames])**power
    freqs = np.fft.rfftfreq(nfft, d=1./fs)
    return S, freqs

def get_bin(xf, f):
    return np.argmin(np.abs(xf - f))

def cepstrogram(frames, fs, nfft=None):
    if not nfft: nfft = frames.shape[-1]
    # calc cepstrum
    S, xf = spectrogram(frames, fs, nfft, power=1.)
    S = util.amp_to_db(S)
    C, _ = spectrogram(S, fs, nfft, power=1.)
    C = util.amp_to_db(C)
    return C


def _dct(S, axis=-1, dct_type=2, norm='ortho'):
    return dct(S, axis=axis, type=dct_type, norm=norm)


def hilbert_test(signal, fs):
    analytic_signal = hilbert(signal)

    amplitude_envelope = np.abs(analytic_signal)
    
    instantaneous_phase = np.unwrap(np.angle(analytic_signal))
    
    instantaneous_frequency = (np.diff(instantaneous_phase) /
    
                               (2.0*np.pi) * fs)
    
    return amplitude_envelope, instantaneous_frequency



