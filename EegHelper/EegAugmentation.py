
from .EegData import EegDataPoint
import mne
from mne.decoding import Scaler
import numpy as np

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