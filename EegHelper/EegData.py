import mne
import numpy as np
from torch.utils.data import Dataset
import torch
import pandas as pd
from tqdm import tqdm

"""
Converts a numpy array of EEG data to an mne object
"""
def np_to_mne(array, ch_names) -> mne.io.RawArray:
    ch_types = ['eeg' for _ in range(len(ch_names))]
    info = mne.create_info(ch_names=ch_names, ch_types=ch_types, sfreq=128)
    raw = mne.io.RawArray(array, info)
    #Generate the epoch object
    return raw

"""
Loads a file as a panda df, if the length is less than 260 return none, else return a numpy array - transposed (channels, timesteps) 
"""
def load_file(path) -> np.array:
    df = pd.read_csv(path, index_col=0) #Read  through panda
    if len(df) < 256:                   #Not enough samples. We want 256
        return None, None
    else:
        df = df.iloc[0:256]             #Return first 256 samples
    return np.array(df).T, list(df.columns)

"""
Takes in a list of files, and how many to load.
Returns a list of all loaded labels, and all EegDataPoints.
"""
def files_to_datapoints(epoc_files, first_n=500, return_nulls:bool = True) -> np.array:
    #First read the mandatory epoc files
    all_points_epoc=[]                  
    all_labels_epoc=set()     
       
    print("Loading epoc data...")            
    for path in tqdm(epoc_files[0:first_n]): 
        result, ch_names = load_file(path)
        path = path.replace('\\', '/') 
        if type(result) is np.ndarray: #
            label = path.split('/')[2] 
            label = label.split('_')[0]
            if not return_nulls and (label == '-1' or label == '10'):
                continue
            
            all_labels_epoc.add(label if label != '-1' else '10')
            all_points_epoc.append(EegDataPoint(result, label if label != '-1' else '10', ch_names=ch_names)) 
    all_points_epoc = np.array(all_points_epoc)
    all_labels_epoc=list(all_labels_epoc)
    
    return all_points_epoc, all_labels_epoc

"""
Saves a list of EegDataPoints to csv files.
"""
def data_points_to_file(points:np.array, path:str):
    id = 0
    for point in tqdm(points):
        columns = point.ch_names
        frame = pd.DataFrame(point.raw_data.T, columns=columns)
        frame.to_csv(f"{path}/{point.label}_{id}.csv")
        id+=1

"""
Stores a data point, which contains raw data, mne object, and label.
"""
class EegDataPoint:

    def __init__(self, data, label, ch_names = ["AF3", "F7", "F3", "FC5", "T7", "P7", "O1", "O2", "P8", "T8", "FC6", "F4", "F8", "AF4"]):
        self.label = label
        self.raw_data = data
        raw = np_to_mne(data, ch_names)
        self.mne_object = raw
        self.ch_names = ch_names
        self.scores = [1 for _ in range(len(self.ch_names))]

    """
    Returns the epoch containing the entire event.
    """
    def __get_full_epoch__(self) -> mne.EpochsArray:
        epoch = mne.EpochsArray(np.expand_dims(self.mne_object._data, axis=0), self.mne_object.info)
        return epoch

    """
    Apply a filter to the data, essentially cutting out the data below 
    l_freq and above h_freq.
    """
    def filter_mne(self, l_freq = 3, h_freq = 30):
        self.mne_object = self.mne_object.filter(
            l_freq = l_freq, 
            h_freq = h_freq,
            picks='eeg'
        )
        self.raw_data = self.mne_object._data
    """
    Re-references the data by subtracting the average signal from all signals.
    """
    def average_reference(self):
        self.mne_object.set_eeg_reference(ref_channels='average')
        self.raw_data = self.mne_object._data
    
    """
    Keeps only the desired channels.
    """
    def crop_to_channels(self, channels:list):
        drop = [channel for channel in self.mne_object.ch_names if channel not in channels]
        self.mne_object.drop_channels(drop)
        self.raw_data = self.mne_object._data
        self.ch_names = self.mne_object.ch_names
    
    """
    Performs cleaning in the following order:
    1.Average reference
    2.Filter
    3.Channel crop
    AVERAGE REFERENCE MUST BE DONE FIRST.
    """
    def full_clean(self, channels = None, l_freq = 3, h_freq = 30):
        if(channels == None):
            channels = self.mne_object.ch_names
        self.average_reference()
        self.filter_mne(l_freq, h_freq)
        self.crop_to_channels(channels)
    
    """
    Uses spline interpolation to clean bad channels. Bad channels are marked outside of the object with other logic. It is based on the correlation coeeficients (scores).
    Note that when interpolating bad channels, normalization should NOT be applied, and it should happen after full clean is run.
    """
    def interpolate_bad_channels(self):
        selection = self.mne_object.ch_names
        mont1020 = mne.channels.make_standard_montage('standard_1020')

        ind = [i for (i, channel) in enumerate(mont1020.ch_names) if channel in selection]
        mont1020_new = mont1020.copy()
        mont1020_new.ch_names = [mont1020.ch_names[x] for x in ind]
        kept_channel_info = [mont1020.dig[x+3] for x in ind]
        mont1020_new.dig = mont1020.dig[0:3]+kept_channel_info
        self.mne_object.set_montage(mont1020)

        self.mne_object = self.mne_object.interpolate_bads(reset_bads=True)
        self.raw_data = self.mne_object._data

"""
Dataset for loading into dataloader.
"""
class EegDataset(Dataset):
    def __init__(self, data_points:np.array, labels:list, transforms = None):
        self.data_points = data_points
        self.labels = labels
        self.transforms = transforms

    def __len__(self) -> int: 
        return len(self.data_points)
    
    def __getitem__(self, i) -> torch.Tensor:
        ans = np.zeros(len(self.labels), dtype=np.int16)
        ans[self.labels.index(self.data_points[i].label)] = 1
        data = self.data_points[i].raw_data

        if self.transforms != None:
            data = self.transforms(data)
        
        return torch.Tensor(data), torch.Tensor(ans)

