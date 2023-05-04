from .EegData import EegDataPoint
import mne
from mne.decoding import Scaler
import numpy as np
import torch.nn as nn
from tqdm import tqdm
import random
import math
import logging

logger = logging.getLogger(__name__)

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
    def get_robust_scaler(self) -> mne.decoding.Scaler:
        scaler = Scaler(with_mean=True, with_std=True, scalings='median')
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

class TimeCut(nn.Module):
    """
    Splits the data at a random location across the time axis, and shifts it left.
    """
    def __init__(self, prob:float=0.5, sigma:float = 0.2):
        super().__init__()
        self.prob = prob
        self.sigma = sigma
    
    def forward(self, data:np.array)->np.array:
        if random.random() < self.prob:
            size = np.random.normal(0,self.sigma)
            cut_size = (data.shape[1]*size)//1
            cut_size = int(cut_size)
            r = np.split(data, [data.shape[1] - cut_size, data.shape[1]], axis=1)
            
            #Shouldnt happen, but failsafe to avoid a crash
            if r[0].shape[1] + r[1].shape[1] != 256:
                logger.warning(f'Invalid chunk size for time shift. r[0] = {r[0].shape[1]}, r[1] = {r[1].shape[1]}. Not applied.')
                return data

            data = np.concatenate((r[1], r[0]), axis = 1)

        return data

class InvertFrequencies(nn.Module):
    """
    Flips the frequencies of the data. f(t) -> -f(t)
    """
    def __init__(self, prob:float=0.5):
        super().__init__()
        self.prob = prob
    
    def forward(self, data:np.array) -> np.array:
        if random.random() < self.prob:
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

class EegSmoothZeroMask(nn.Module):
    """
    Smoothly flattens the signal to 0 across a range defined by the curve. Performs at a random location.
    """
    def __init__(self, window:int = 10, prob:float=0.5):
        super().__init__()
        self.window = window
        self.prob = prob
    
    def __f__(self, x:int, window:int)->float:
        return 1/(1+pow(math.e, x-window))

    def __g__(self, x:int)->float:
        return 1/(1+pow(math.e, -x))

    def __masktransformation__(self, x:int, location:int = 0)->float:
        return -self.__f__(x-location-5, self.window) - self.__g__(x-location-5) + 2
    
    def forward(self, data):
        if random.random() < self.prob:
            mask = np.ones(shape = data.shape[1])
            location = random.randrange(-5, 220)
            for x in range(mask.shape[0]):
                mask[x] = self.__masktransformation__(x, location)

            return data * np.array([mask for _ in range(data.shape[0])])
        
        return data
