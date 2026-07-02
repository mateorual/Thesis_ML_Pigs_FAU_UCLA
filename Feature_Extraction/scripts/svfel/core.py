from svfel.utils import io, util
from svfel.features import cepstral, stats, fit, entropy

import numpy as np
import pandas as pd



class VowelLoader():
        
    def __init__(self):
        self.fs = None
        self.wav = None
        self.cycles = None
        self.bits = None
        

    def load(self, wav_file, cycle_file):
        self.fs, wav = io.read_wav(wav_file)
        self.wav, self.bits = io.pcm_to_float(wav)
        self.cycles = io.read_cycles(cycle_file)
        assert self.fs == 44100
        assert self.bits == 16
    
    
    def reset(self):
        self.__init__()



class SVFEx():
    # Sustained Vowel Feature Extractor
    
    
    def __init__(self):
        self.fs = None
        self.wav = None
        self.cycles = None
        
        # mfcc
        self.S_mfcc, self.mfcc, self.dmfcc, self.ddmfcc = None, None, None, None
      
        
    def load(self, wav_file, cycle_file):
        self.fs, wav = io.read_wav(wav_file)
        self.wav, self.bits = io.pcm_to_float(wav)
        self.cycles = io.read_cycles(cycle_file)
        
        
    def extract_mfcc(self, s):
        '''
        Calculate MFCCs, Delta-MFCCs and Delta-Delta-MFCCs of audio sequence.

        Parameters
        ----------
        s : dict
            Settings.
        '''
        
        # normalize
        wav = util.mean_center(self.wav, -1)
        wav = util.maxabs_scale(self.wav, -1)   
        # pre-emphasis?
        
        # frames
        framegen = util.FrameGenerator(wav, self.fs)
        framegen.set(s['frame_size'], s['stride'], s['unit'], s['wfunc'], s['pad'])
        
        # mfcc
        self.mfcc, _ = cepstral.mfcc(framegen, self.fs, s['nfft'], 
                                     s['nmfcc'], s['nfilt'], s['lifter'], 
                                     s['fmin'], s['fmax'], s['s_norm'])
        
        # delta
        self.dmfcc = cepstral.delta_cc(self.mfcc, 1)
        self.ddmfcc = cepstral.delta_cc(self.mfcc, 2)
            
        
    def time_aggregate(self, feature, s):
        '''
        Calculate aggregating features for a feature time series.

        Parameters
        ----------
        feature : 2d-array
            (n_time, n_feat).
        s : list
            Aggregates to compute.

        Returns
        -------
        aggregate : TYPE
            DESCRIPTION.

        '''
        # transpose so time axis = -1
        feature = feature.T

        values, names = [], []

        ### statistical features
        if 'mean' in s:
            values.append(stats.xmean(feature))
            names.append('mean')
            
        if 'std' in s:
            values.append(stats.xstd(feature))
            names.append('std')
            
        if 'skew' in s:
            values.append(stats.xskew(feature))
            names.append('skew')
            
        if 'kurt' in s:
            values.append(stats.xkurt(feature))
            names.append('kurt')
            
        if 'mean_abs_dev' in s:
            values.append(stats.xmean_abs_dev(feature))
            names.append('mean_abs_dev')
            
        if 'median_abs_dev' in s:
            values.append(stats.xmedian_abs_dev(feature))
            names.append('median_abs_dev')
            
        if 'median' in s:
            values.append(stats.xmedian(feature))
            names.append('median')
            
        if 'q1' in s:
            values.append(stats.xquantile(feature, 0.25))
            names.append('q1')
            
        if 'q3' in s:
            values.append(stats.xquantile(feature, 0.75))
            names.append('q3')
            
        if 'iqr' in s:
            values.append(stats.xiqr(feature, [0.25, 0.75]))
            names.append('iqr')
            
        if 'max' in s:
            values.append(stats.xmax(feature))
            names.append('max')
            
        if 'min' in s:
            values.append(stats.xmin(feature))
            names.append('min')
            
        if 'range' in s:
            values.append(stats.xrange(feature))
            names.append('range')
            
        ### fit features
        if 'linear_slope' in s:
            x = np.arange(feature.shape[-1])/self.fs
            coef = fit.least_squares(feature, x, order=1)
            values.append(coef[:,0])
            names.append('linear_slope')
            
        if 'linear_rmse' in s:
            x = np.arange(feature.shape[-1])/self.fs
            coef = fit.least_squares(feature, x, order=1)
            y_hat = fit.draw(x, coef).T
            values.append(fit.rmse(feature, y_hat))
            names.append('linear_rmse')
        
        # ### entropy features 
        # # !!! check settings
        # if s['apen']:
        #     values.append(entropy.approximate_entropy(feature))
        #     names.append('apen')
            
        # if s['sampen']:
        #     values.append(entropy.sample_entropy(feature))
        #     names.append('sampen')
            
        # if s['svden']:
        #     values.append(entropy.svd_entropy(feature))
        #     names.append('svden')
            
        # if s['permen']:
        #     values.append(entropy.permutation_entropy(feature))
        #     names.append('permen')
            
        aggregate = pd.DataFrame(values, index=names).T
        return aggregate
    
    def reset(self):
        self.__init__()
        
 