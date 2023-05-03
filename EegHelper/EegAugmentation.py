
from .EegData import EegDataPoint
import mne
from mne.decoding import Scaler
import numpy as np
import torch.nn as nn

"""
Class for normalizing EEG data using the MNE Scaler object.
In the background this is sklearns StandardScalar across the channel axis.
"""
class NormalizationHelper:
    def __init__(self, data_points:list):
        self.data_points = data_points

    def __get_all_epochs__(self) -> list:
        epochs = []
        for point in self.data_points:
            epoch = point.__get_full_epoch__()._data
            epochs.append(epoch.reshape(epoch.shape[1], epoch.shape[2]))
        epochs = np.array(epochs)
        return epochs
    
    """
    Returns an mne scalar object fit with the points given in the constructor.
    """
    def get_standard_scaler(self) -> mne.decoding.Scaler:
        scaler = Scaler(with_mean=True, with_std=True, scalings='mean')
        scaler.fit(self.__get_all_epochs__())
        return scaler
    
    """
    Returns a list of transformed points.
    """
    @staticmethod
    def fit_points(scaler, data_points) -> np.array:
        new_points = []
        for point in data_points:
            data = scaler.transform(point.__get_full_epoch__()._data)
            data = data.reshape(data.shape[1], data.shape[2])
            new_points.append(EegDataPoint(
                data,
                point.label,
                ch_names=point.ch_names
            ))
        return np.array(new_points)

#TODO Time domain flip , f(t) = -f(t), time reverse and smooth time mask most effective. Fourier transform surrogate. Try a random lowpass/highpass filter. Random channel dropout
 
class EegGaussianNoise(nn.Module):
    """
    Adds random gaussian noise to an EEG signal
    """
    def __init__(self, mu = 0, sigma = 1):
        super().__init__()
        self.mu = mu
        self.sigma = sigma
    
    def forward(self, data):
        s = np.random.normal(self.mu, self.sigma, size=data.shape)
        return data + s
    
    def __repr__(self):
        return f"Gaussian distribution: \nMean {self.mu} \nStandard devistion {self.sigma}"

class EegRandomScaling(nn.Module):
    """
    Multiplies the signal by a random constant value
    """
    def __init__(self, mu = 1, sigma = 0.1):
        super().__init__()
        self.mu = mu
        self.sigma = sigma

    def forward(self, data):
        s = np.random.normal(self.mu, self.sigma, size=1)
        return data*s
    
    def __repr__(self):
        return f"Gaussian distribution for selection of random scalar: \nMean {self.mu} \nStandard devistion {self.sigma}"
    
