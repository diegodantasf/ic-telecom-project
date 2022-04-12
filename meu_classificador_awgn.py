import sys
import os

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import pandas as pd
import numpy as np

from commpy.channels import SISOFlatChannel

sys.path.append(os.path.abspath('ml4comm'))
from ml4comm.ml4comm.qam_awgn import generate_symbols

np.random.seed(2) # Important

def generate_dataset():
    M            = 16      # QAM modulation
    num_symbols  = 6000    # Number of transmitted symbols
    SNR_dB       = 15      # Signal to noise ratio in dB     
    code_rate    = 1       # Rate of the used code
    Es           = 1       # Average symbol energy

    # Generate the QAM symbols
    symbs, indices = generate_symbols(num_symbols, M)

    channel = SISOFlatChannel(None, (1 + 0j, 0j))
    channel.set_SNR_dB(SNR_dB, float(code_rate), Es)
    channel_output = channel.propagate(symbs)

    # Train
    train_size = int(0.5*len(indices))
    y_train = indices[:train_size]
    X_train = np.stack([np.real(channel_output[:train_size]),
                        np.imag(channel_output[:train_size])], axis=1)

    # Test
    y_test = indices[train_size:]
    X_test = np.stack([np.real(channel_output[train_size:]),
                    np.imag(channel_output[train_size:])], axis=1)

    return [(X_train, y_train), (X_test, y_test)]

# get n samples from dataset (assume dataset already shuffled)
def get_n_samples(dataset_train, n):
    return (dataset_train[0][:n], dataset_train[1][:n])

def main():
    dataset_train, dataset_test = generate_dataset() # Dataset shuffled
    dataset_test_10 = get_n_samples(dataset_train, 10)

if __name__ == '__main__':
    main()
