import numpy as np
from scipy.signal import freqz, firwin, filtfilt, resample_poly
from scipy.stats import gaussian_kde
from scipy.stats import norm
from math import gcd
# from scipy.interpolate import interp1d

def digital_filter_response(num, den):
    # get filter response
    w, h = freqz(num, den) # frequency (rad/sample), frequency response (complex)
    return w, h  


def delta(feature, ndiff=1, axis=0):
    return np.diff(feature, ndiff, axis)


def xlogx(X, base=2):
    # https://github.com/raphaelvallat/antropy/tree/master/antropy
    # https://github.com/mne-tools/mne-features/tree/399092ab015995afc66453ecc99fbe3e502f9a9a/mne_features
    """Returns x * log_base(x) if x is positive, 0 if x == 0, and np.nan
    otherwise. This handles the case when the power spectrum density
    takes any zero value.
    """
    xlx = np.zeros(X.shape)
    xlx[X < 0] = np.nan   
    xlx[X > 0] = X[X > 0] * np.log(X[X > 0]) / np.log(base)
    return xlx

def pre_emphasize(frames, a=0.97):
    # pre-emphasis (amplify high frequencies)
    return np.hstack([frames[:, 0].reshape(-1,1), frames[:, 1:] - a * frames[:, :-1]])

def mean_center(x, axis=None):
    return x - np.mean(x, axis)

def minmax_norm(x, axis=None):
    return (x - np.min(x, axis, keepdims=True)) / (np.max(x, axis, keepdims=True) - np.min(x, axis, keepdims=True))

def minmax_scale(x, xmin, xmax, axis=None):
    x_norm = (x - np.min(x, axis)) / (np.max(x, axis) - np.min(x, axis))
    return x_norm * (xmax - xmin) + xmin

def zscore_norm(x, axis=None):
    return (x - np.mean(x, axis)) / np.std(x, axis)

def maxabs_scale(x, axis=None):
    return x / np.max(np.abs(x), axis)


def pow_to_db(x, xref=1.0, xmin=1e-12):
    # power to decibel conversion
    # x_db = 10*log10(X/Xref) = 10*log10(X) - 10*log10(Xref)
    x_db = 10.0 * np.log10(np.maximum(xmin, x))
    x_db -= 10.0 * np.log10(np.maximum(xmin, xref))
    return x_db

def amp_to_db(x, xref=1.0, xmin=1e-12):
    # amplitude to decibel conversion
    # x_db = 20*log10(X/Xref) = 10*log(X**2/Xref)
    x = np.square(x)
    return pow_to_db(x, xref, xmin)

def rad_to_hz(x, fs):
    # convert rad to Hz
    return x * fs / (2*np.pi)

def hz_to_mel(f):
    return (2595. * np.log10(1. + f/700.))
    
def mel_to_hz(z):
    return (700. * (10.**(z/2595.) - 1.))

def hz_to_bark(f):
    return 6. * np.arcsinh(f / 600.)
    
def bark_to_hz(z):
    return 600. * np.sinh(z / 6.)


class FrameGenerator:
    # Iterator
    # add padding (for signal/frames)?

    def __init__(self, signal, fs=None):
        # signal
        self.fs = fs
        self.signal = signal # !!! add signal_ so signal doesnt have to be overwritten by pad
        self.signal_length = signal.shape[-1]
        # default settings
        self.frame_size = None
        self.stride = None
        self.wfunc = None
        self.pad_len = None
        self.n_frames = 0
        self.tc = None
        if self.fs:
            self.t = np.linspace(0, signal.shape[-1]/fs, signal.shape[-1])
    
    
    def _get_wfunc(self, wfunc):
        # if wfunc is defined
        if wfunc: 
            # if wfunc is a string
            if type(wfunc) == str:
                # get & return function
                if wfunc == 'hamming':
                    return np.hamming
                elif wfunc == 'hanning':
                    return np.hanning
                elif wfunc == 'blackman':
                    return np.blackman
                else:
                    raise ValueError('Unknown window function.')
            # return function otherwise
            return wfunc
        # return square window otherwise
        else: 
            return np.ones
        
            
    def set(self, frame_size, stride, unit='samples', wfunc=None, pad=None):
        # settings
        if unit == 'samples':
            self.frame_size = int(frame_size)
            self.stride = int(stride)
        elif unit == 'seconds':
            self.frame_size = int(frame_size * self.fs)
            self.stride = int(stride * self.fs)
        
        wfunc = self._get_wfunc(wfunc)
        self.wfunc = wfunc(self.frame_size).astype(self.signal.dtype)
        
        # if pad_len: self.pad_len = int(pad_len * self.fs)
        ###
        if pad == 'end':
            print('WARNING: padding overwrites internal signal!')
            self.n_frames = int(np.ceil((len(self.signal) - self.frame_size) / self.stride) + 1)
            new_len = int((self.n_frames-1)*self.stride+self.frame_size)
            signal = np.zeros(new_len, dtype=np.float64)
            signal[:self.signal_length] = self.signal
            self.signal = signal
            self.signal_length = new_len
        else:
            # get frame indices (cuts off remainder)
            assert self.frame_size <= len(self.signal)
            self.n_frames = int((len(self.signal) - self.frame_size) / self.stride) + 1
        self.idxs = np.tile(np.arange(0, self.frame_size, dtype=np.int32), (self.n_frames, 1)) + np.tile(np.arange(0, self.n_frames * self.stride, self.stride, dtype=np.int32), (self.frame_size, 1)).T
        if self.fs:
            # get frame center time
            self.tc = np.stack([(self.frame_size/2 + self.stride*i)/self.fs for i in range(self.n_frames)])
        
    def __iter__(self):
        self.i = 0
        return self


    def __next__(self):
        if self.i < self.n_frames:
            i = self.i
            self.i += 1
            return self.signal[self.idxs[i]] * self.wfunc
        else:
            raise StopIteration
            
    def get_all(self):
        #return [n for n in self]
        return self.signal[self.idxs] * self.wfunc
    
    def split(self, signal, apply_wfunc=False):
        #return [n for n in self]
        if apply_wfunc:
            return signal[self.idxs] * self.wfunc
        else:
            return signal[self.idxs]
    
    
def nextpow2(n):
    # https://github.com/cournape/talkbox/blob/master/scikits/talkbox/tools/correlations.py
    """Return the next power of 2 such as 2^p >= n.
    Notes
    -----
    Infinite and nan are left untouched, negative values are not allowed."""
    if np.any(n < 0):
        raise ValueError("n should be > 0")

    if np.isscalar(n):
        f, p = np.frexp(n)
        if f == 0.5:
            return p-1
        elif np.isfinite(f):
            return p
        else:
            return f
    else:
        f, p = np.frexp(n)
        res = f
        bet = np.isfinite(f)
        exa = (f == 0.5)
        res[bet] = p[bet]
        res[exa] = p[exa] - 1
        return res
    
    
def create_xx(features):
    # https://github.com/fraunhoferportugal/tsfel/blob/21b023aae69fa1a1ff9abe3c534980fa7883025f/tsfel/
    """Computes the range of features amplitude for the probability density function calculus.
    Parameters
    ----------
    features : nd-array
        Input features
    Returns
    -------
    nd-array
        range of features amplitude
    """

    features_ = np.copy(features)

    if max(features_) < 0:
        max_f = - max(features_)
        min_f = min(features_)
    else:
        min_f = min(features_)
        max_f = max(features_)

    if min(features_) == max(features_):
        xx = np.linspace(min_f, min_f + 10, len(features_))
    else:
        xx = np.linspace(min_f, max_f, len(features_))

    return xx

def kde(features):
    # https://github.com/fraunhoferportugal/tsfel/blob/21b023aae69fa1a1ff9abe3c534980fa7883025f/tsfel/
    """Computes the probability density function of the input signal using a Gaussian KDE (Kernel Density Estimate)
    Parameters
    ----------
    features : nd-array
        Input from which probability density function is computed
    Returns
    -------
    nd-array
        probability density values
    """
    features_ = np.copy(features)
    xx = create_xx(features_)

    if min(features_) == max(features_):
        noise = np.random.randn(len(features_)) * 0.0001
        features_ = np.copy(features_ + noise)

    kernel = gaussian_kde(features_, bw_method='silverman')

    return np.array(kernel(xx) / np.sum(kernel(xx)))


def gaussian(features):
    # https://github.com/fraunhoferportugal/tsfel/blob/21b023aae69fa1a1ff9abe3c534980fa7883025f/tsfel/
    """Computes the probability density function of the input signal using a Gaussian function
    Parameters
    ----------
    features : nd-array
        Input from which probability density function is computed
    Returns
    -------
    nd-array
        probability density values
    """

    features_ = np.copy(features)

    xx = create_xx(features_)
    std_value = np.std(features_)
    mean_value = np.mean(features_)

    if std_value == 0:
        return 0.0
    pdf_gauss = norm.pdf(xx, mean_value, std_value)

    return np.array(pdf_gauss / np.sum(pdf_gauss))

def cosine_similarity(X, Y):
    # https://en.wikipedia.org/wiki/Cosine_similarity
    num = np.sum(X * Y, axis=-1)
    den = np.linalg.norm(X, axis=-1) * np.linalg.norm(Y, axis=-1)
    return num/den


def fir_filt(frames, fs, numtaps=None, cutoffs=[0,1000], window=('kaiser', 8.6)):
    if numtaps is None:
        numtaps = int((fs/50)*(80/22))
        if numtaps%2 == 0: 
            numtaps += 1
    # create filter   
    flt = firwin(numtaps=numtaps, cutoff=cutoffs, window=window, pass_zero='bandpass', fs=fs)
    # apply filter
    filtered = []
    for frame in frames: 
        filtered.append(filtfilt(flt, 1, frame)) # zero phase shift
    return np.stack(filtered)


def fir_lowpass(frames, fs, numtaps=None, cutoff=5000, window=('kaiser', 8.6)):
    if numtaps is None:
        numtaps = int((fs/50)*(80/22))
        if numtaps%2 == 0: 
            numtaps += 1
    # create filter   
    flt = firwin(numtaps=numtaps, cutoff=cutoff, window=window, pass_zero='lowpass', fs=fs)
    # apply filter
    filtered = []
    for frame in frames: 
        filtered.append(filtfilt(flt, 1, frame)) # zero phase shift
    return np.stack(filtered)


def fir_highpass(frames, fs, numtaps=None, cutoff=50, window=('kaiser', 8.6)):
    if numtaps is None:
        numtaps = int((fs/50)*(80/22))
        if numtaps%2 == 0: 
            numtaps += 1
    # create filter   
    flt = firwin(numtaps=numtaps, cutoff=cutoff, window=window, pass_zero='highpass', fs=fs)
    # apply filter
    filtered = []
    for frame in frames: 
        filtered.append(filtfilt(flt, 1, frame)) # zero phase shift
    return np.stack(filtered)



def cycle_split(signal, cycles):
    # 1d signal, 2d cycles
    
    # check cycles
    assert ((cycles[1:, 0] - cycles[:-1, 1]) == 1).all()
    # define split
    split = np.append(cycles[:,0], [cycles[-1,1]+1], axis=0)
    # split = split-1
    # split signal
    csignal = np.split(signal, split, axis=0)[1:-1]
    return csignal


def spread_cycle_values(values, cycles, signal_length):
    # !!! shape of value
    # !!! what use cases?
    signal = np.zeros(signal_length)
    
    for val, cycle in zip(values, cycles):
        signal[cycle[0]:cycle[1]+1] = val
    
    signal[0:cycles[0,0]] = val[0]
    signal[cycles[-1,1]:signal_length] = val[-1]
    
    # n_smooth = np.mean(np.diff(cycles[:,0]))*5
   
    # for i in range(signal.shape[0]):
    #     signal[i] = np.mean(signal[max(0, i-int(n_smooth/2)):min(signal_length, i+int(n_smooth/2)+1)], axis=0)
        
    return signal

def mean_smooth(frames, n=0):
    # frames = (n_frame, n_val)
    # smooth over last axis
    frames_ = np.copy(frames)
    if n > 0:
        for i in range(frames.shape[-1]):
            start = max(0, i-int(n/2))
            stop =  min(frames.shape[-1], i+int(n/2)+1)
            frames_[:, i] = np.mean(frames[:, start:stop], axis=-1)
    return frames_


def resample_sinc(y, fs, fs_re):
    """
    Interpolates y, sampled with fs
    Output y_re is sampled at fs_re instants  
    """
    x = np.linspace(0, y.shape[0], int(fs*(y.shape[0]/fs))) #!!!
    x_re = np.linspace(0, y.shape[0], fs_re)

    # period    
    T = x[1] - x[0]
    
    sincM = np.tile(x_re, (len(x), 1)) - np.tile(x[:, np.newaxis], (1, len(x_re)))
    y_re = np.dot(y, np.sinc(sincM/T))
    return y_re

def resample_polyphase(y, fs, fs_re):
    """
    Rational resampling using scipy.signal.resample_poly.
    Tested and better version than resample_sinc that uses interpolation.
    To debug, you can plot PSD(Welch) for original and resampled signal and
    verify that harmonic/formant peaks are preserved
    """

    if fs == fs_re:
        return y.astype(np.float64, copy=False)

    g = gcd(fs, fs_re)
    up = fs_re // g
    down = fs // g
    y = resample_poly(y, up, down, padtype="constant")
    return y.astype(np.float64, copy=False)


def recycle(cycles, duration, fs_old, fs_new):
    l_old = int(duration*fs_old)
    t_old = np.arange(l_old)/fs_old
    l_new = int(duration*fs_new)
    t_new = np.arange(l_new)/fs_new
    cycles_new = np.zeros_like(cycles)

    # Safety: clip cycle indices to valid range for t_old array
    # This prevents "index out of bounds" errors when cycles extend beyond signal
    cycles_clipped = np.clip(cycles, 0, l_old - 1)

    for i, c in enumerate(cycles_clipped):
        t = t_old[c[0]]
        cycles_new[i,0] = np.argmin(np.abs(t_new-t))
    cycles_new[:-1,1] = cycles_new[1:,0]-1
    # end of last cycle
    t = t_old[cycles_clipped[-1,1]]
    cycles_new[-1,1] = np.argmin(np.abs(t_new-t))
    return cycles_new