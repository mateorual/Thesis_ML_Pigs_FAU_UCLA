from svfel.utils import util, tft
from svfel.features import temporal
import numpy as np
from scipy.signal import butter, filtfilt
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt


def spectral_flatness_gat(S):
    # spectral flatness like gat
    sf = []
    for s in S:
        # geometric mean
        gm = 10./(len(s)-1) * np.sum(np.log10(np.square(s[1:])), keepdims=True)     
        # arithmetic mean
        am = 10.*np.log10(1./(len(s)-1) * np.sum(np.square(s[1:]), keepdims=True))
        sf.append(gm-am)
    sf = np.stack(sf)
    return sf


def _cosine_similarity(X, Y):
    # https://en.wikipedia.org/wiki/Cosine_similarity
    num = np.sum(X * Y, axis=-1)
    den = np.linalg.norm(X, axis=-1) * np.linalg.norm(Y, axis=-1)
    return num/den


def wmc(cframes, signal=None):
    # waveform matching coefficient
    if signal is None:
        signal = np.hstack(cframes)
    mean_cycle_len = np.mean([len(c) for c in cframes])
    # divide into equal parts
    n = int((len(signal) / mean_cycle_len)) # number of possible equal parts
    fg = util.FrameGenerator(signal)
    fg.set(n, n)
    mc = fg.get_all()
    # get cosine similarity of subsequent cycles
    frames_x = mc[:-1, :]
    frames_y = mc[1:, :]    
    wmc = _cosine_similarity(frames_x, frames_y)    
    return wmc.reshape(-1,1)

def hnr(cframes):
    ### harmonics-to-noise ratio

    # pad all cycles to maximum length using nan
    # (GAT pads cycles with mean value, result is distorted)
    max_cycle_len = np.max([len(cframe) for cframe in cframes])    
    # pad with nan
    cframes_ = []
    for cframe in cframes:
        cframes_.append(np.hstack([cframe, np.zeros(max_cycle_len-len(cframe))+np.nan]))
    cframes_ = np.stack(cframes_)

    # average cycle
    avg_cycle = np.nanmean(cframes_, axis=0)
    # harmonics
    h = len(cframes_) * np.sum(np.square(avg_cycle))
    # noise
    n = np.sum([np.sum(np.square(avg_cycle[:len(cframe)] - cframe)) for cframe in cframes])
    # hnr
    hnr = 10.*np.log10(h/n)
    return np.array([[hnr]], dtype=np.float64)



def _get_harmonics(s, xf, f0, f_tol=5):
    # max number of harmonics
    n_max = int(xf.max()/f0)
    harmonics = []
    for n in range(1, n_max+1):
        # define search range
        min_bin = tft.get_bin(xf, n*f0-f_tol)
        max_bin = tft.get_bin(xf, n*f0+f_tol)
        # append idx of max magnitude
        harmonics.append(np.argmax(s[min_bin:max_bin+1]) + min_bin)
    return np.stack(harmonics)
        

def _get_subharmonics(s, xf, f0, f_tol=5):
    subharmonics = []
    # search for 2 subharmonics (alt: search while bin changes and f>bin_res)
    for n in [1/2]:
        # define search range
        min_bin = tft.get_bin(xf, n*f0-f_tol)
        max_bin = tft.get_bin(xf, n*f0+f_tol)
        # append idx of max magnitude
        subharmonics.append(np.argmax(s[min_bin:max_bin+1]) + min_bin)
    return np.stack(subharmonics)[::-1]


def harmonics_intensity(S, xf, f0, f_tol=10, pthres=None, debug=False):
    
    hi = [] 
    for s in S:
        # get harmonic idxs
        harmonics = _get_harmonics(s, xf, f0, f_tol)
        f0_idx = harmonics[0]
        # get subharmonic idxs
        subharmonics = _get_subharmonics(s, xf, f0, f_tol)
        # combine
        harmonics = np.hstack((subharmonics, harmonics))
        # magnitude threshold condition
        if pthres is not None:
            harmonics = harmonics[s[harmonics]>=pthres*s[f0_idx]]
        
        # hi.append(100. * (np.sum(s[harmonics], keepdims=True)-s[harmonics[0]]) / (np.sum(s, keepdims=True))) # gat
        hi.append(100. * np.sum(s[harmonics], keepdims=True) / np.sum(s, keepdims=True))
    
        if debug:
            plt.figure(figsize=(12,8))
            plt.plot(xf, s, label=hi[-1])
            plt.scatter(xf[harmonics], s[harmonics], c='r')
            plt.legend()
            plt.show()
    
    hi = np.stack(hi) 
    return hi


def _noise_energy(s, harmonics, nfft, frame_len):
    ### noise energy
    noise = np.zeros(len(s))
    
    ### get peak regions Pi (harmonic peak +/- width)
    # define harmonic half width
    width = int(round(2*nfft/frame_len))
    pi_start, pi_end = [], []
    for h in harmonics:
        start = np.max([0, h-width])
        end = np.min([len(s), h+width+1])
        pi_start.append(start)
        pi_end.append(end)
    
    ### fill dip regions Di (regions between peak regions)
    missing, exclude = 0, 0
    di_start, di_end = [], []   
    # (add range before first Pi and after last Pi)
    for start, end in zip(pi_start+[len(s)], [1]+pi_end):
        # if peak regions overlap -> dip region doesn't exist
        if end-start>=0: 
            # if first dip region: add 0
            if len(di_start) < 1:
                di_start.append(0)
                di_end.append(1)
            # else: copy previous dip region
            else:
                di_start.append(di_start[-1])
                di_end.append(di_end[-1])
            missing+=1
        # if dip region exists
        else:
            # copy spectrum within dip region
            noise[end:start] = np.square(s[end:start])
            di_start.append(end)
            di_end.append(start)
            missing=0
        # if two successive dip regions don't exist -> exclude this frame
        if missing > 1: 
            exclude=1
            break

    ### fill peak regions Pi
    if exclude==0:       
        for i,(start,end) in enumerate(zip(pi_start, pi_end)):
            # fill peak region with mean of two surrounding dip regions
            # dip 1 - peak 1 - dip 2 --> peak 1 = sqrt(mean(mean(dip 1**2), mean(dip 2**2)))
            val = .5* (np.sum(np.square(s[di_start[i]:di_end[i]])) / len(s[di_start[i]:di_end[i]]) +
                       np.sum(np.square(s[di_start[i+1]:di_end[i+1]])) / len(s[di_start[i+1]:di_end[i+1]]))      
            noise[start:end] = val

    # don't know why this is separate
    noise[0] = np.square(s[0])
            
    return noise, exclude


def nne_gat(signal, cycles, fs, debug=False):
    # !!! version with fix frames?
    if signal.ndim > 1:
        signal = signal.squeeze()

    # Safety: clip cycle boundaries to signal length
    # This prevents accessing indices beyond the signal array
    cycles = cycles.copy()  # Don't modify original
    cycles = np.clip(cycles, 0, len(signal) - 1)

    ### settings
    # search range -> median period
    search_size = int(0.04 * fs) # 40ms
    search_stride = int(0.02 * fs) # 20ms
    # number of periods -> window size
    n_periods = 10 # !!! different from GAT (7)
    # padding
    max_len = np.max([np.diff(cycles[:,0])])
    pad_len = np.max([int(0.1024*fs), max_len*n_periods+1])
    # harmonic threshold & tolerance
    pthres = 0.05 # GAT=0.05
    ftol = 20
    # frequency range to consider
    f_min = 50
    f_max = np.min((5000, fs/2))
    
    
    noise_energy, signal_energy = [], []
    
    search_start = 0
    search_end = search_size
    while search_end <= signal.shape[-1]:
        
        # get all cycles with starting point in 40ms window
        cycles_in_search = [c for c in cycles[:,0] if c>=search_start and c<=search_end]
        # get median cycle duration
        if len(cycles_in_search) < 2:
            # Not enough cycles in search window, skip to next window
            search_start += search_stride
            search_end += search_stride
            continue
        median_period = int(np.median(np.diff(cycles_in_search)))  
        # define frame of length 7*p_median
        frame_start = search_start
        frame_end = frame_start + n_periods * median_period + 1
        if frame_end > signal.shape[-1]: 
            break
        
        # get frame
        frame = signal[frame_start:frame_end]
        # apply hamming window
        frame = frame * np.hamming(len(frame))
        # zero pad
        frame_pad = np.zeros(pad_len)
        frame_pad[:len(frame)] = frame
        
        # fft
        s, xf = tft.rfft(frame_pad, fs)
        # get f0
        f0 = fs / float(median_period)
        
        ### get harmonics
        # get harmonic idxs
        harmonics = _get_harmonics(s, xf, f0, ftol)
        subharmonics = _get_subharmonics(s, xf, f0, ftol)
        f0_idx = harmonics[0]
        # combine
        harmonics = np.hstack((subharmonics, harmonics))
        # magnitude threshold condition
        harmonics = harmonics[s[harmonics]>=pthres*s[f0_idx]]
        
        # noise energy, energy
        p_noise, exclude = _noise_energy(s, harmonics, len(frame_pad), len(frame))
        p = np.square(s)
        
        # if usable
        if exclude == 0:
            # define frequency range
            min_bin = tft.get_bin(xf, f_min)
            max_bin = tft.get_bin(xf, f_max)
            # get signal and noise energy in range
            signal_energy.append(np.sum(p[min_bin:max_bin]))
            noise_energy.append(np.sum(p_noise[min_bin:max_bin]))

        ### debug
        if debug:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(12,12))
            plt.subplot(221)
            plt.plot(xf, s)
            plt.scatter(xf[harmonics], s[harmonics], c='r')
            plt.subplot(222)
            plt.plot(xf, s)
            plt.scatter(xf[harmonics], s[harmonics], c='r')
            plt.xlim([0,1000])
            plt.subplot(223)
            plt.plot(xf, s**2)
            plt.plot(xf, p_noise)
            plt.subplot(224)
            plt.plot(xf, s**2)
            plt.plot(xf, p_noise)
            plt.xlim([0,1000])
            plt.ylim([0,10])
            plt.show()
        
        # update search window
        search_start += search_stride
        search_end += search_stride
        
    # nne = (p_noise / p_signal)
    nne = util.pow_to_db(np.mean(noise_energy) / np.mean(signal_energy))
    return np.array(nne).reshape(1,1)
    

def gaussian(m, sigma):
    # sigma restricted
    sigma = np.max([sigma, .5])
    # signal range
    x = np.arange(m)
    # exponent
    exp = .5 * np.square((x - (m+1)/2) / (sigma*(m-1)/2))
    # return window = e^(-exp)
    return np.exp(-exp)


def snr_v1_gat(signal, cycles, fs, debug=False):
    if signal.ndim > 1:
        signal = signal.squeeze()

    # Safety: clip cycle boundaries to signal length
    # This prevents accessing indices beyond the signal array
    cycles = cycles.copy()  # Don't modify original
    cycles = np.clip(cycles, 0, len(signal) - 1)

    ### settings
    # window
    frame_size = int(0.160 * fs) # 160ms
    frame_stride = int(0.02 * fs) # 20ms
    pad_len = int(0.320 * fs) # 320 ms
    # fft upscaling
    n_up = 4 
    # harmonic tolerance
    ftol = 20
    # pthres = 0
    # filter
    # order=4, cutoff = .25*fs / (fs/2) == .5
    b, a = butter(N=4, Wn=.5, btype='low')
    # valleys
    fstep = fs / (pad_len * n_up) 
    wmax = int(np.round(12./fstep)) # 24Hz width
    wmin = int(np.round(6./fstep)) # 12Hz width
    h_tol = 10**(12/20) # 12dB
    
    snrv1 = []
    
    frame_start = 0
    frame_end = frame_size    
    while frame_end <= len(signal):
        
        # get 160ms window
        frame = signal[frame_start:frame_end]
        # get f0 estimate
        cycles_in_frame = [c for c in cycles[:,0] if c>=frame_start and c<=frame_end]
        f0 = fs / np.mean(np.diff(cycles_in_frame))
        # gaussian window
        frame = frame * gaussian(len(frame), sigma=.4)
        # pad to 320ms
        frame_pad = np.hstack([frame, np.zeros(pad_len-len(frame))])
        
        # fft
        s, xf = tft.rfft(frame_pad, fs)
        # upscale
        upfunc = interp1d(xf, s, kind='linear')
        xf_up = np.linspace(xf[0], xf[-1], (len(xf)-1)*n_up+1)
        s_up = upfunc(xf_up)
        # apply low pass filter
        s_filt = s_up#filtfilt(b, a, s_up) # !!!
        
        # find harmonics, subharmonics
        harmonics = _get_harmonics(s_filt, xf_up, f0, ftol)
        subharmonics = _get_subharmonics(s_filt, xf_up, f0, ftol)
        # combine
        f0_idx = harmonics[0]
        harmonics = np.hstack((subharmonics, harmonics))

        # Safety: clip all harmonic indices to valid array bounds
        harmonics = harmonics[harmonics < len(s_filt)]
        if len(harmonics) == 0:
            # No valid harmonics, skip this frame
            frame_start += frame_stride
            frame_end += frame_stride
            continue
        f0_idx = harmonics[0]  # Update f0_idx after clipping

        # magnitude threshold condition -> not done in GAT
        # harmonics = harmonics[s_filt[harmonics]>=pthres*s_filt[f0_idx]]
        # noise magnitude
        
        ### noise magnitude approximation
        # get mean magnitude between harmonic peaks +/- width
        noise_mag = []
        # first valley (12Hz -> H-12Hz)
        start_idx = max(wmax, 0)
        end_idx = min(harmonics[0]-wmax+1, len(s_filt))
        if end_idx > start_idx:
            noise_mag.append(np.mean(s_filt[start_idx:end_idx]))
        else:
            noise_mag.append(0.0)
        # inbetween (H1+12Hz -> H2-12Hz)
        for i in range(harmonics.shape[0]-1):
            start_idx = harmonics[i]+wmax
            end_idx = min(harmonics[i+1]-wmax+1, len(s_filt))
            if end_idx > start_idx and start_idx < len(s_filt):
                noise_mag.append(np.mean(s_filt[start_idx:end_idx]))
            else:
                noise_mag.append(0.0)
        # last valley (H+12Hz -> H+F0-12Hz) - fixed bounds checking
        start_idx = harmonics[-1]+wmax
        end_idx = min(harmonics[-1]+harmonics[0]-wmax+1, len(s_filt))
        if end_idx > start_idx and start_idx < len(s_filt):
            noise_mag.append(np.mean(s_filt[start_idx:end_idx]))
        elif start_idx < len(s_filt):
            # If end goes beyond bounds, just use what's available
            noise_mag.append(np.mean(s_filt[start_idx:]))
        
        ### synthetic harmonic signal
        # only keep harmonics that are 12dB above surrounding noise
        harmonics_ = []
        for i, h in enumerate(harmonics):
            # Skip if harmonic index is out of bounds
            if h >= len(s_filt):
                continue
            h_mag = s_filt[h]
            if i+2 <= len(noise_mag):
                n_mag = np.max(noise_mag[i:i+2])
            else:
                n_mag = noise_mag[i] if i < len(noise_mag) else 0
            if h_mag > h_tol * n_mag:
                harmonics_.append(h)
        # at least keep f0
        if len(harmonics_) < 1:
            if f0_idx < len(s_filt):
                harmonics_ = [f0_idx]
            else:
                # Skip this frame if f0 is out of bounds
                frame_start += frame_stride
                frame_end += frame_stride
                continue
                
        
        ### get harmonic bandwidths
        bw = []
        # for each harmonic
        for h in harmonics_:
            # magnitude of harmonic
            h_mag = s_filt[h]
            # mask where magnnitude of s < 0.5*h_mag
            mask = s_filt<(.5*h_mag)
            # get mask left and right of harmonic with bounds checking
            mask_l_start = max(0, h-wmax)
            mask_r_end = min(len(mask), h+wmax)
            mask_r = mask[h:mask_r_end]
            mask_l = mask[mask_l_start:h]
            try:
                # get first index below 0.5*h_mag
                l = mask_l_start + np.where(mask_l==1)[0][-1] + 1
                r = h + np.where(mask_r==1)[0][0] - 1
            except IndexError:
                # just take +/-12Hz with bounds checking
                l = max(0, h-wmax)
                r = min(len(s_filt)-1, h+wmax)
                ## original: remove harmonics with bw outside of [12,24Hz]
                # should technically be excluded!!!
                # if left/right half-value not found
                # if (~l.any()) or (~r.any()):
            bw.append(r-l)

        # Safety check: ensure we have valid bandwidths
        if len(bw) == 0:
            frame_start += frame_stride
            frame_end += frame_stride
            continue

        # get B0
        h_magnitudes = [s_filt[h] for h in harmonics_ if h < len(s_filt)]
        if len(h_magnitudes) == 0:
            frame_start += frame_stride
            frame_end += frame_stride
            continue
        b0 = bw[np.where(h_magnitudes==np.max(h_magnitudes))[0][0]]

        ### approximate pure harmonic spectrum
        hwgaus = int(np.round(.25*f0/fstep)) # should be .5*B0???
        sigma = b0 / 2.354820045030949 # conversion of fwhm to sigma
        mu = 0.
        gausmax = 1./(sigma*np.sqrt(2.*np.pi)) * np.exp(-.5*np.square(-mu)/np.square(sigma))
        k = np.arange(-hwgaus, hwgaus+1)
        gaus =  1./(sigma*np.sqrt(2.*np.pi)) * np.exp(-.5*np.square(k-mu)/np.square(sigma)) / gausmax
        h_gaus = np.zeros_like(s_filt)
        for h in harmonics_:
            # Skip if h is out of bounds
            if h >= len(s_filt):
                continue
            mag = s_filt[h]
            # Clip indices to valid array bounds (fix for low F0 like pig vocalizations)
            start_idx = max(0, h - hwgaus)
            end_idx = min(len(h_gaus), h + hwgaus + 1)
            gaus_start = hwgaus - (h - start_idx)
            gaus_end = gaus_start + (end_idx - start_idx)
            h_gaus[start_idx:end_idx] = gaus[gaus_start:gaus_end] * mag
        
        ### total energy, harmonic energy
        et = np.sum(np.square(s_filt))
        eh = np.sum(np.square(h_gaus))
        
        if eh<et:
            snr = 10.*np.log10(et/(et-eh))
            snr = np.min([snr, 36]) # 36db defined as max
            snrv1.append(snr)
        else:  
            # harmonic magnitude mustn't be > total magnitude
            # in original algorithm bw is adjusted
            # snrv1.append(np.nan)
            
            snrv1.append(36) #!!! don't know dude

        ### debug
        if debug:
            # import matplotlib.pyplot as plt
            plt.figure(figsize=(12,12))
            plt.subplot(321)
            plt.plot(xf, s)
            plt.plot(xf_up, s_up)
            plt.subplot(322)
            plt.plot(xf_up, s_up)
            plt.plot(xf_up, s_filt)
            plt.subplot(323)
            plt.plot(xf_up, s_filt)
            plt.scatter(xf_up[harmonics], s_filt[harmonics], c='r')
            plt.subplot(324)
            plt.plot(xf_up, s_filt)
            plt.scatter(xf_up[harmonics], s_filt[harmonics], c='r')
            plt.xlim([0,1500])
            plt.subplot(325)
            plt.plot(s_filt, linewidth=3)
            plt.plot(h_gaus, linestyle='--')
            plt.subplot(326)
            plt.plot(s_filt, linewidth=3)
            plt.plot(h_gaus, linestyle='--')
            plt.xlim([0,500])
            plt.show()
        # update window
        frame_start += frame_stride
        frame_end += frame_stride
        
    if len(snrv1) < 1: 
        return None
    else:
        return np.stack(snrv1).reshape(-1,1)
### SNR v1/v2
### GNE











