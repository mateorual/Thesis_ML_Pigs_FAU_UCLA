from svfel.utils import util

import numpy as np
from sklearn.neighbors import KDTree



### each entropy feature iterates over axis 0 -> summarizes axis 1


def _delay_embedding(x, dim=3, delay=1):
    # https://github.com/raphaelvallat/antropy/tree/master/antropy
    # https://github.com/mne-tools/mne-features/tree/399092ab015995afc66453ecc99fbe3e502f9a9a/mne_features
    # https://github.com/neuropsychology/NeuroKit/blob/master/neurokit2/complexity/complexity_delay_embeddingding.py
    """Time-delay embedding.
    """
    x = np.asarray(x)
    N = x.shape[-1]
    assert x.ndim in [1, 2], "Only 1D or 2D arrays are currently supported."
    if dim * delay > N: raise ValueError("Error: dim * delay should be lower than x.size")
    if delay < 1: raise ValueError("delay has to be at least 1.")
    if dim < 2: raise ValueError("dim has to be at least 2.")


    # 1D array (n_times)
    Y = np.zeros((dim, N - (dim - 1) * delay))
    for i in range(dim):
        Y[i] = x[(i * delay):(i * delay + Y.shape[1])]
    return Y.T


def permutation_entropy(frames, dim=3, delay=1, normalize=False):
    # https://github.com/raphaelvallat/antropy/tree/master/antropy
    # https://github.com/mne-tools/mne-features/tree/399092ab015995afc66453ecc99fbe3e502f9a9a/mne_features
    """Permutation Entropy.
    Parameters
    ----------
    frames: n_frames, n_features
    x : list or np.array
        One-dimensional time series of shape (n_times)
    dim : int
        dim of permutation entropy. Default is 3.
    delay : int, list, np.ndarray or range
        Time delay (lag). Default is 1. If multiple values are passed
        (e.g. [1, 2, 3]), AntroPy will calculate the average permutation
        entropy across all these delays.
    normalize : bool
        If True, divide by log2(dim!) to normalize the entropy between 0
        and 1. Otherwise, return the permutation entropy in bit.
    Returns
    -------
    pe : float
        Permutation Entropy.
    """
    # # If multiple delay are passed, return the average across all d
    # if isinstance(delay, (list, np.ndarray, range)):
    #     return np.mean([permutation_entropy(x, dim=dim, delay=d,
    #                     normalize=normalize) for d in delay])
    
    if frames.ndim == 1: 
        frames.reshape(1, -1)
    
    permen = []
    for f in frames:
        ran_dim = range(dim)
        hashmult = np.power(dim, ran_dim)
        assert delay > 0, "delay must be greater than zero."
        # Embed x and sort the dim of permutations
        sorted_idx = _delay_embedding(f, dim=dim, delay=delay).argsort(kind='quicksort')
        # Associate unique integer to each permutations
        hashval = (np.multiply(sorted_idx, hashmult)).sum(1)
        # Return the counts
        _, c = np.unique(hashval, return_counts=True)
        p = c / c.sum()
        pe = -util.xlogx(p).sum()
        if normalize:
            pe /= np.log2(np.math.factorial(dim))
        permen.append(pe)
    return np.stack(permen)


def svd_entropy(frames, dim=3, delay=1, normalize=False):
    # https://github.com/raphaelvallat/antropy/tree/master/antropy
    # https://github.com/mne-tools/mne-features/tree/399092ab015995afc66453ecc99fbe3e502f9a9a/mne_features
    """Singular Value Decomposition entropy.
    Parameters
    ----------
    x : list or np.array
        One-dimensional time series of shape (n_times)
    dim : int
        dim of SVD entropy (= length of the embedding dimension).
        Default is 3.
    delay : int
        Time delay (lag). Default is 1.
    normalize : bool
        If True, divide by log2(dim!) to normalize the entropy between 0
        and 1. Otherwise, return the permutation entropy in bit.
    Returns
    -------
    svd_e : float
        SVD Entropy
    """
    if frames.ndim == 1: 
        frames.reshape(1, -1)
    svden = []
    for f in frames:
        mat = _delay_embedding(f, dim=dim, delay=delay)
        W = np.linalg.svd(mat, compute_uv=False)
        # Normalize the singular values
        W /= sum(W)
        svd_e = -util.xlogx(W).sum()
        if normalize:
            svd_e /= np.log2(dim)
        svden.append(svd_e)
    return np.stack(svden)


def approximate_entropy(frames, dim=2, metric='chebyshev'):
    # https://github.com/raphaelvallat/antropy/tree/master/antropy
    # https://github.com/mne-tools/mne-features/tree/399092ab015995afc66453ecc99fbe3e502f9a9a/mne_features
    """Approximate Entropy.
    Parameters
    ----------
    x : list or np.array
        One-dimensional time series of shape (n_times).
    dim : int
        Embedding dimension. Default is 2.
    metric : str
        distance metric function
    Returns
    -------
    ae : float
        Approximate Entropy.
    """
    
    _all_metrics = KDTree.valid_metrics
    if metric not in _all_metrics:
        raise ValueError('The given metric (%s) is not valid. The valid '
                         'metric names are: %s' % (metric, _all_metrics))
    
    
    
    if frames.ndim == 1: 
        frames.reshape(1, -1)
    
    apen = []
    for f in frames:
    
        r = 0.2 * np.std(f, ddof=0)    
    
        # compute phi(dim, r)
        emb_data1 = _delay_embedding(f, dim, 1)
        count1 = KDTree(emb_data1, metric=metric).query_radius(emb_data1, r,
                                                               count_only=True
                                                               ).astype(np.float64)
        # compute phi(dim + 1, r)
        emb_data2 = _delay_embedding(f, dim + 1, 1)
        count2 = KDTree(emb_data2, metric=metric).query_radius(emb_data2, r,
                                                               count_only=True
                                                               ).astype(np.float64)
    
        phi_1 = np.mean(np.log(count1 / emb_data1.shape[0]))
        phi_2 = np.mean(np.log(count2 / emb_data2.shape[0]))
        
        apen.append(phi_1 - phi_2)
        
    return np.stack(apen)


def sample_entropy(frames, dim=2, metric='chebyshev'):
    # https://github.com/raphaelvallat/antropy/tree/master/antropy
    # https://github.com/mne-tools/mne-features/tree/399092ab015995afc66453ecc99fbe3e502f9a9a/mne_features
    """Sample Entropy.
    Parameters
    ----------
    x : list or np.array
        One-dimensional time series of shape (n_times).
    dim : int
        Embedding dimension. Default is 2.
    metric : str
        distance metric function
    Returns
    -------
    se : float
        Sample Entropy.
    """
    
    _all_metrics = KDTree.valid_metrics
    if metric not in _all_metrics:
        raise ValueError('The given metric (%s) is not valid. The valid '
                         'metric names are: %s' % (metric, _all_metrics))
    
    if frames.ndim == 1: 
        frames.reshape(1, -1)
    
    sampen = []
    for f in frames:
    
        r = 0.2 * np.std(f) 

        # compute phi(dim, r)
        emb_data_1 = _delay_embedding(f, dim, 1)[:-1]
        count_1 = KDTree(emb_data_1, metric=metric).query_radius(emb_data_1, r,
                                                               count_only=True
                                                               ).astype(np.float64)
        # compute phi(dim + 1, r)
        emb_data_2 = _delay_embedding(f, dim + 1, 1)
        count_2 = KDTree(emb_data_2, metric=metric).query_radius(emb_data_2, r,
                                                               count_only=True
                                                               ).astype(np.float64)
    
        phi_1 = np.mean((count_1 - 1) / (emb_data_1.shape[0] - 1))
        phi_2 = np.mean((count_2 - 1) / (emb_data_2.shape[0] - 1))
        
        # if np.allclose(phi[:, 0], 0) or np.allclose(phi[:, 1], 0):
        # raise ValueError('Sample Entropy is not defined.')
        
        sampen.append(-np.log(phi_2/phi_1))
    
    return np.stack(sampen)
