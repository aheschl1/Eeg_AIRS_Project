from .EegData import EegDataPoint
import mne
from mne.decoding import Scaler
import numpy as np
import torch.nn as nn
from tqdm import tqdm
import random
from braindecode.augmentation import FTSurrogate

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
        for point in tqdm(data_points):
            data = scaler.transform(point.__get_full_epoch__()._data)
            data = data.reshape(data.shape[1], data.shape[2])
            new_points.append(EegDataPoint(
                data,
                point.label,
                ch_names=point.ch_names
            ))
        return np.array(new_points)

#TODO smooth time mask, time reverse and smooth time mask most effective. Fourier transform surrogate. Try a random lowpass/highpass filter. Random channel dropout

class FurrierSurrogate(nn.Module):
    """
    Applied a Furrier transform surrogate with the braindecode.augmentation library
    """
    def __init__(self, prob:float=0.5, phase_noise_magnitude:float=1):
        super().__init__()
        assert False, "Don't use this doesn't work"
        assert(phase_noise_magnitude >= 0 and phase_noise_magnitude <= 1)
        self.prob = prob
        self.phase_noise_magnitude = phase_noise_magnitude
    
    def forward(self, data:np.array)->np.array:
        return FTSurrogate(probability=self.prob, phase_noise_magnitude=self.phase_noise_magnitude)(data)


class InvertFrequencies(nn.Module):
    """
    Flips the frequencies of the data. f(t) -> -f(t)
    """
    def __init__(self, prob:float=0.5):
        super().__init__()
        self.prob = prob
    
    def forward(self, data:np.array) -> np.array:
        if random() < self.prob:
            return -1*data
        return data
    
    def __repr__(self):
        return super().__repr__()

class TimeDomainFlip(nn.Module):
    """
    FLips the time domain so that {x0, x1, x2, ......, xn} -> {xn, x n-1, x n-2, ......, x0}
    """
    def __init__(self, prob:float=0.5):
        super().__init__()
        self.prob = prob
    
    """
    Slips a given numpy array along axis 1
    """
    def __flipped__(self, data:np.array) -> np.array:
        return np.flip(data, axis=1).copy()

    def forward(self, data:np.array) -> np.array:
        if random.random() < self.prob:
            return self.__flipped__(data)
        return data
    
    def __repr__(self):
        return super().__repr__()

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
        return super().__repr__()

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
        return super().__repr__()
    
