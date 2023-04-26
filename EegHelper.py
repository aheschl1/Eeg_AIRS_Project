import mne
import numpy as np
from torch.utils.data import Dataset
import torch

"""
Converts a numpy array of EEG data to an mne object
"""
def np_to_mne(array):
    ch_types = ['eeg' for _ in range(5)]
    ch_names=['AF3', 'AF4', 'T7', 'T8', 'Pz']
    info = mne.create_info(ch_names=ch_names, ch_types=ch_types, sfreq=128)
    raw = mne.io.RawArray(array, info)
    return raw

"""
Stores a data point, which contains raw data, mne object, and label
"""
class EegDataPoint:

    def __init__(self, data, label):
        self.label = label
        self.raw_data = (data.T)
        raw = np_to_mne(data)
        self.mne_object = raw

    def clean(self, l_freq, h_freq):
        #Definetly needs improvement
        self.mne_object = self.mne_object.filter(l_freq = l_freq, h_freq = h_freq,
            picks='eeg',
            method="fir",
            phase="zero-double",
            fir_design="firwin",
            pad="edge")
        self.raw_data = self.mne_object._data.T

"""
Dataset for loading
"""
class EegDataset(Dataset):
    def __init__(self, data_points:np.array, labels:list):
        self.data_points = data_points
        self.labels = labels

    def __len__(self):
        return len(self.data_points)
    
    def __getitem__(self, i):
        label = np.zeros((len(self.labels), 1), dtype=np.float32)
        label[self.labels.index(self.data_points[i].label)] = 1.0
        return torch.Tensor(self.data_points[i].raw_data), torch.Tensor(label)