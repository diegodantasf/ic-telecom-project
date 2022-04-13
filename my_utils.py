import numpy as np
import sys
import os
from commpy.channels import SISOFlatChannel

sys.path.append(os.path.abspath('ml4comm'))
from ml4comm.qam_awgn import generate_symbols
from ml4comm.qam_crazy import crazy_channel_propagate
from ml4comm.qam_analyzer import plot_decision_boundary, ser, plot_confusion_matrix

from sklearn.model_selection import GridSearchCV

class Dataset:
  def __init__(self, channel_type='awgn'):
    self.M            = 16      # QAM modulation
    self.num_symbols  = 6000    # Number of transmitted symbols
    self.SNR_dB       = 15      # Signal to noise ratio in dB     
    self.code_rate    = 1       # Rate of the used code
    self.Es           = 1       # Average symbol energy
  
    # Generate the QAM symbols
    symbs, indices = generate_symbols(self.num_symbols, self.M)
    
    if channel_type == 'awgn':
      channel = SISOFlatChannel(None, (1 + 0j, 0j))
      channel.set_SNR_dB(self.SNR_dB, float(self.code_rate), self.Es)
      channel_output = channel.propagate(symbs)
    elif channel_type == 'crazy':
      channel_output = crazy_channel_propagate(symbs, self.SNR_dB)
    else:
      raise Exception("Channel type must be 'awgn' or 'crazy'")
      
  
    # Train
    self.train_size = int(0.5*len(indices))
    self.y_train = indices[:self.train_size]
    self.X_train = np.stack([np.real(channel_output[:self.train_size]),
                        np.imag(channel_output[:self.train_size])], axis=1)
  
    # Test
    self.y_test = indices[self.train_size:]
    self.X_test = np.stack([np.real(channel_output[self.train_size:]),
      np.imag(channel_output[self.train_size:])], axis=1)
  
  def get_train_dataset(self, n_samples=3000):
    if (n_samples > 3000):
      raise Exception("Max size is 3000")
    
    return (self.X_train[:n_samples], self.y_train[:n_samples])
    
    
  def get_test_dataset(self):
    return (self.X_test, self.y_test)

def grid_search(model, X_train, y_train, parameters, name='MODEL NAME'):
    clf = GridSearchCV(model, parameters, scoring='neg_mean_squared_error')
    clf.fit(X_train, y_train)

    results = clf.cv_results_
    print(f'------------ {name} ---------------')
    print(f'TRAIN SIZE: {len(y_train)}')
    ranks = results['rank_test_score']
    print(f'ranks: {ranks}')
    for rank in results['rank_test_score']:
        params = results['params'][rank-1]
        score = results['mean_test_score'][rank-1]
        print(f'rank {rank}: {params} - score: {score}')

    return clf.best_estimator_

def test_model(model, X_test, y_test, name='MODEL NAME'):
    model_ser = ser(model, X_test, y_test)
    print(f'------------- {name} -------------')
    print(f'{name}:\n SER:\t {model_ser:.2%}')
    print('\n')

def plots(model, X_data, y_data, n_classes):
    plot_decision_boundary(model, X_data, y_data, legend=True)
    plot_confusion_matrix(model, X_data, y_data, n_classes)


  