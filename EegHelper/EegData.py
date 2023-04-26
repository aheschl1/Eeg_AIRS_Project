import mne
import numpy as np
from torch.utils.data import Dataset
import torch
import pandas as pd

"""
Converts a numpy array of EEG data to an mne object
"""
def np_to_mne(array):
    ch_names = ["AF3", "F7", "F3", "FC5", "T7", "P7", "O1", "O2", "P8", "T8", "FC6", "F4", "F8", "AF4"]
    ch_types = ['eeg' for _ in range(len(ch_names))]
    info = mne.create_info(ch_names=ch_names, ch_types=ch_types, sfreq=128)
    raw = mne.io.RawArray(array, info)
    return raw

"""
Loads a file as a panda df, if the length is less than 260 return none, else return a numpy array - transposed (channels, timesteps) 
"""
def load_file(path):
    df = pd.read_csv(path, index_col=0) #Read  through panda
    if len(df) < 260:                   #Not enough samples. We want 260
        return None
    else:
        df = df.iloc[0:260]             #Return first 320 samples
    return np.array(df).T

"""
Takes in a list of files, and how many to load.
Returns a list of all loaded labels, and all EegDataPoints.
"""
def files_to_datapoints(files, first_n=500):
    all_points=[]                  
    all_labels=set()                    
    for path in files[0:first_n]: 
        result = load_file(path)
        path = path.replace('\\', '/') 
        if type(result) is np.ndarray: #
            label = path.split('/')[2] 
            label = label.split('_')[0]
            all_labels.add(label)
            all_points.append(EegDataPoint(result, label)) 

    all_points = np.array(all_points)
    all_labels=list(all_labels)

    return all_points, all_labels

"""
Stores a data point, which contains raw data, mne object, and label.
"""
class EegDataPoint:

    def __init__(self, data, label):
        self.label = label
        self.raw_data = (data.T)
        raw = np_to_mne(data)
        self.mne_object = raw

    def filter_mne(self, l_freq, h_freq):
        #Definetly needs improvement
        self.mne_object = self.mne_object.filter(l_freq = l_freq, h_freq = h_freq,
            picks='eeg',
            method="fir",
            phase="zero-double",
            fir_design="firwin",
            pad="edge")
        self.raw_data = self.mne_object._data.T


"""
Dataset for loading into dataloader.
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

"""
dataset = EegDataset(data_points=all_points, labels=all_labels)
train, test = train_test_split(dataset, train_size=0.8, shuffle=True)

train_dataloader = DataLoader(train, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test, batch_size=batch_size, shuffle=True)

"""