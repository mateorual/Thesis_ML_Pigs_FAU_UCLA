import os
import numpy as np
from scipy.io import wavfile
import struct

def read_wav(filepath):
    # read fs, wav
    fs, wav = wavfile.read(filepath)
    return fs, wav


def write_wav(filepath, wav, fs):
    # make dir
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    # write wav
    wavfile.write(filepath, fs, wav)
    
    
def pcm_to_float(x):
    # scale to -1.0 -- 1.0
    if x.dtype == 'int16':
        nb_bits = 16  # -> 16-bit wav files
    elif x.dtype == 'int32':
        nb_bits = 32  # -> 32-bit wav files
    max_nb_bit = float(2 ** (nb_bits - 1))       
    x_ = x / (max_nb_bit - 1)  # samples is a numpy array of floats representing the samples 
    return x_.astype(np.float64), nb_bits


def float_to_pcm(x, bits):
    # convert float to pcm
    max_val = float(2 ** (bits - 1))
    x_pcm = np.array([round(xi*(max_val-1)) for xi in x], dtype=np.int16)
    return x_pcm


def read_cycles(path):
    '''
    Parameters
    ----------
    path : string
        Filepath.

    Returns
    -------
    cycles : np.array
        Array of size (n_cycles x 2).
        Col1: start point of cycle
        Col2: end point of cycle
        
    '''  
    if not '.cycles' in path: path += '.cycles'
    if os.path.isfile(path):
        cycles=[]
        with open(path, 'rb') as f:
            n_cycles = struct.unpack('I', bytearray(f.read(4)))[0] # number of cycles
            f.read(4) # discard 2
            for i in range(n_cycles):
                start = int(struct.unpack('I', bytearray(f.read(4)))[0])  # cycle start point
                end = int(struct.unpack('I', bytearray(f.read(4)))[0])  # cycle end point
                cycles.append([start, end])
            cycles=np.vstack(cycles)
        return cycles
    else:
        print(f'{path} NOT FOUND.')

        
def write_cycles(cs, ce, path):
    n_cycles = len(cs)
    
    with open(path, 'wb') as file:
        # number of cycles
        file.write(struct.pack('I', n_cycles))
        # 2 (number of columns, like GAT)
        file.write(struct.pack('I', 2))
        for i in range(n_cycles):
            # start point
            file.write(struct.pack('I', cs[i]))
            # end point
            file.write(struct.pack('I', ce[i]))
    
    