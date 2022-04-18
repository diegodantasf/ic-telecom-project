import numpy as np
import sys
import os
from commpy.channels import SISOFlatChannel

sys.path.append(os.path.abspath('ml4comm'))
from ml4comm.qam_awgn import generate_symbols
from ml4comm.qam_crazy import crazy_channel_propagate
from ml4comm.qam_analyzer import plot_decision_boundary, ser, plot_confusion_matrix

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.svm import SVC

from imblearn.under_sampling import RandomUnderSampler

import pandas as pd

class Dataset:
  def __init__(self, channel_type='awgn'):
    self.M            = 16      # QAM modulation
    self.num_symbols  = 6000    # Number of transmitted symbols
    self.SNR_dB       = 15      # Signal to noise ratio in dB     
    self.code_rate    = 1       # Rate of the used code
    self.Es           = 1       # Average symbol energy

    np.random.seed(52)
    
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
    if n_samples > 3000:
      raise Exception("Max size is 3000")
    if n_samples // self.M < 5:
      raise Exception("Less than 5 samples for each class, set n_samples = self.M * 5 at least")
    
    rus = RandomUnderSampler(sampling_strategy = {c:n_samples//self.M for c in range(self.M)}, random_state=52)
    try:
      return rus.fit_resample(self.X_train, self.y_train)
    except ValueError:
      return (self.X_train[:n_samples], self.y_train[:n_samples])
    
  def get_test_dataset(self):
    return (self.X_test, self.y_test)

def SER(y_true, y_pred):
    return np.sum(y_true != y_pred) / len(y_true)

def grid_search(model, X_train, y_train, parameters, name='MODEL NAME'):
    clf = GridSearchCV(model, parameters, scoring=make_scorer(SER, greater_is_better=False), verbose=1)
    clf.fit(X_train, y_train)

    print(f'------------ {name} ---------------')
    print(f'TRAIN SIZE: {len(y_train)}')
    
    df = pd.DataFrame(clf.cv_results_)
    df = df.loc[:, ['params', 'mean_test_score', 'rank_test_score']]
    df = df.sort_values(by=['rank_test_score'])

    for i, row in df.iterrows():
        print(f"rank {row['rank_test_score']}: {row['params']} - SER: {-row['mean_test_score']:.3%}")

    return clf.best_estimator_

def test_model(model, X_test, y_test, name='MODEL NAME'):
    model_ser = ser(model, X_test, y_test)
    print(f'------------- {name} -------------')
    print(f'{name}:\n SER:\t {model_ser:.2%}')
    print('\n')

def plots(model, X_data, y_data, n_classes):
    plot_decision_boundary(model, X_data, y_data, legend=True)
    plot_confusion_matrix(model, X_data, y_data, n_classes)
    
def get_classifier(name):
    classifiers = {
        'decision_tree': {
            'model': DecisionTreeClassifier(),
            'parameters': {
                'max_depth': [100, 50, 10]
            }
        },
        'knn': {
            'model': KNeighborsClassifier(),
            'parameters': {
                'n_neighbors': [1, 2, 3, 4, 5]
            }
        },
        'random_forest': {
            'model': RandomForestClassifier(),
            'parameters': {
                'n_estimators': [4, 8, 16],
                'criterion': ['gini', 'entropy'],
                'max_depth': [3, 4, 5, 6],
                'max_features': [2],
                'random_state': [52]
            }
        },
        'svm_rbf': {
            'model': SVC(),
            'parameters': {
                'C':[1e-2, 1e-1, 1, 1e2,1e3, 1e4,1e5, 1e6, 1e7, 1e8, 1e9],
                #'degree':[3, 4],
                'gamma':['scale', 'auto', 1e-2, 1e-3, 1e-4, 1e-5],
                'random_state': [52],
                #'tol':[1e-4, 1e-3],
                #'max_iter':[1e5, 1e4],
                'decision_function_shape':['ovo']
            }
        }
    }
    
    return (classifiers[name]['model'], classifiers[name]['parameters'])
  
def evaluate_model(name, channel='awgn', train_sizes=[500, 250, 80]):
    clf, parameters = get_classifier(name)
    
    ds = Dataset(channel_type=channel)
    
    for size in train_sizes:
        X_train, y_train = ds.get_train_dataset(n_samples=size)
        X_test, y_test = ds.get_test_dataset()
        best_clf = grid_search(clf, X_train, y_train, parameters, name)
        plots(best_clf, X_test, y_test, ds.M)
        test_model(best_clf, X_test, y_test, name)


if __name__ == '__main__':
    evaluate_model(name='svm_rbf', channel='awgn', train_sizes=[720, 640, 480, 320,160, 144])
