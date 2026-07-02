from svfel.utils import util
import numpy as np


def apply(S, fbank):
    # filterbank x spectrogram
    return np.dot(S, fbank.T)


def mel_filterbank(fs, nfft=4096, nfilt=128, fmin=0.0, fmax=None, dtype=np.float64, debug=False):
    
    # define maximum frequency
    if fmax is None:
        fmax = float(fs) / 2
       
    # convert range
    mel_min = util.hz_to_mel(fmin)
    mel_max = util.hz_to_mel(fmax)
    # center freqs of mel bands - equally spaced between mel_min & mel_max
    fc = util.mel_to_hz(np.linspace(mel_min, mel_max, int(nfilt)+2))
    # center freqs of FFT bins
    fftfreqs = np.fft.rfftfreq(n=nfft, d=1./fs)
    
    # create triangular filters
    fdiff = np.diff(fc)
    ramps = np.subtract.outer(fc, fftfreqs)

    filterbank = []
    for i in range(nfilt):
        # lower and upper slopes for all bins
        lower = -ramps[i] / fdiff[i]
        upper = ramps[i + 2] / fdiff[i + 1]
        # intersect them with each other and zero
        filter = np.maximum(0., np.minimum(lower, upper))
        filterbank.append(filter)
        
    filterbank = np.array(filterbank, dtype=dtype)
    
    # slaney-style mel is scaled to be approx constant energy per channel
    # enorm = 2.0 / (fc[2 : nfilt + 2] - fc[:nfilt])
    # filterbank *= enorm[:, np.newaxis]
    # other normalization -> sum(channel) = 1
    filterbank = filterbank / np.sum(filterbank, axis=1, keepdims=True)
    
    # plot
    if debug:
        import matplotlib.pyplot as plt
        plt.figure()
        for f in filterbank:
            plt.plot(fftfreqs, f)
            # plt.vlines(fc, 0, filterbank.max())
            # plt.xlim([0,500])
        plt.title('Mel Filterbank')
        plt.show()
    
    return filterbank, fc[1:-1]


def bark_filterbank(fs, nfft=4096, nfilts=24, width=1., fmin=0.0, fmax=None, dtype=np.float64,  debug=False):
    
    # define maximum frequency
    if fmax == None:
        fmax = float(fs/2)
    # convert range    
    bark_min = util.hz_to_bark(fmin)
    bark_max = util.hz_to_bark(fmax)
    
    # n filters
    if nfilts == None:
        nfilts = int(np.ceil(bark_max - bark_min) + 1)
        
    # bark per filt
    bark_step = (bark_max - bark_min) / (nfilts - 1)

    # Frequency of each FFT bin in Bark
    fftfreqs = np.fft.rfftfreq(nfft, d=1./fs)
    bark_bin = util.hz_to_bark(fftfreqs)
    
    # fbank different formula as in paper
    fc = []
    filterbank = []
    for i in range(nfilts):
        mid = bark_min + i * bark_step # center freq
        # Linear slopes in log-space (i.e. dB) intersect to trapezoidal window
        lo = (bark_bin - mid - 0.5)
        hi = (bark_bin - mid + 0.5)
        filter = 10 ** (np.minimum(np.zeros_like(hi), np.minimum(hi, -2.5*lo) / width))
        fc.append(util.bark_to_hz(mid))
        filterbank.append(filter)
        
    filterbank = np.array(filterbank, dtype=dtype)
    
    # normalization -> sum(channel) = 1
    filterbank = filterbank / np.sum(filterbank, axis=1).reshape(-1,1)
    
    if debug:
        import matplotlib.pyplot as plt
        plt.figure()
        for f in filterbank:
            plt.plot(fftfreqs, f)
        plt.title('Bark Filterbank')
        plt.show()

    return filterbank, np.stack(fc)


def gammatone_filterbank(fs, nfft=4096, nfilt=64, fmin=0.0, fmax=None, width=1., erbtype='glasberg', debug=False):
    # https://github.com/detly/gammatone/tree/master/gammatone
    # https://spafe.readthedocs.io/en/latest/features/gfcc.html
    # DOI:10.1109/TMM.2012.2199972
    
    if fmax == None:
        fmax = float(fs) / 2.
        
    flen = len(np.fft.rfftfreq(nfft, d=1./fs))
    
    # erb filter models
    # DOI:10.1109/TMM.2012.2199972
    if erbtype == 'glasberg':
        # Glasberg and Moore Parameters
        ear_q = 9.26449 
        min_bw = 24.7
        order = 1
    elif erbtype == 'greenwood':
        # Greenwood
        ear_q = 7.23
        min_bw = 22.85
        order = 1
    elif erbtype == 'lyon':
        # Lyon
        ear_q = 8.
        min_bw = 125.
        order = 2
    else: 
        raise ValueError

    # center frequencies
    # DOI:10.1109/TMM.2012.2199972
    xi = np.arange(1, nfilt+1)
    em = ear_q * min_bw
    step = (ear_q / nfilt) * np.log((fmax + em) / (fmin + em))
    fc = (fmax + em) * np.exp((-xi) * step / ear_q) - em
    fc = fc[::-1]
    
    # equivalent rectangular bandwidths
    # DOI:10.1109/TMM.2012.2199972
    erb = width*((fc / ear_q) ** order + min_bw ** order) ** ( 1 / order)
    
    ### magical gammatone filterbank creation
    # gammatone function: https://en.wikipedia.org/wiki/Gammatone_filter
    T = 1 / fs                      # sampling period
    B = 1.019 * 2. * np.pi * erb    # duration of impulse response
    arg = 2 * np.pi * fc * T        # w
    common = -T * np.exp(-(B * T))  
    rt_pos = np.sqrt(3 + 2 ** 1.5)  
    rt_neg = np.sqrt(3 - 2 ** 1.5)  
    
    k11 = np.cos(arg) + rt_pos * np.sin(arg)
    k12 = np.cos(arg) - rt_pos * np.sin(arg)
    k13 = np.cos(arg) + rt_neg * np.sin(arg)
    k14 = np.cos(arg) - rt_neg * np.sin(arg)
    
    A11 = common * k11
    A12 = common * k12
    A13 = common * k13
    A14 = common * k14
    
    A11, A12, A13, A14 = A11[..., np.newaxis], A12[..., np.newaxis], \
                         A13[..., np.newaxis], A14[..., np.newaxis]
    
    vec = np.exp(2j * arg)
    gain_arg = np.exp(1j * arg - B * T)
    gain = np.abs(  (vec - gain_arg * k11)
                  * (vec - gain_arg * k12)
                  * (vec - gain_arg * k13)
                  * (vec - gain_arg * k14)
                  * (  T * np.exp(B * T)
                  / (-1. / np.exp(B * T) + 1. + vec * (1. - np.exp(B * T)))
                  )**4)
        
    B2 = np.exp(-2. * B * T)
    r = np.sqrt(B2)
    theta = 2. * np.pi * fc * T
    pole = (r * np.exp(1j * theta))[..., np.newaxis]
    
    gt_order = 4
    ucirc = np.exp(1j * 2. * np.pi * np.arange(0, flen) / nfft)[np.newaxis, ...]
    
    filterbank = (  np.abs(ucirc + A11 * fs) * np.abs(ucirc + A12 * fs)
                  * np.abs(ucirc + A13 * fs) * np.abs(ucirc + A14 * fs)
                  * np.abs(fs * (pole - ucirc) * (pole.conj() - ucirc)) ** (-gt_order)
                  / gain[..., np.newaxis])
    
    # normalization???
    # each area of the squared frequency response normalized to be unity (pncc, kim et al)
    filterbank = filterbank / np.sum(filterbank, axis=1).reshape(-1,1) 
    
    # modified like pncc: all values < 0.5% * channel-max are set to 0
    # implementation here
    
    if debug:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10,6))
        xf = np.fft.rfftfreq(nfft, d=1./fs)
        # plt.subplot(121)
        for f in filterbank: 
            plt.plot(xf, f)
            plt.vlines(fc, 0, filterbank.max())
            plt.ylabel('amplitude')
        plt.xlim([0,50])
        plt.subplot(122)
        for f in filterbank: 
            plt.plot(xf, 20*np.log(f))
        plt.ylabel('power dB')
        # plt.xlim([0,50])
        plt.suptitle(f'Gammatone Filterbank {erbtype}')
        # plt.savefig('y.png')
        plt.show()
    
    fc[fc<0] = 0.
    return filterbank[:, :flen], fc

