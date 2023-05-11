import pandas as pd


def get_id_mapping(report_path:str='./MindBigData-Imagenet-IN/WordReport-v1.04.txt'):
    id_mapping={}
    with open(report_path) as file:
        for line in file:
            name = line.split()[0]
            id = line.split()[2]
            id_mapping[id] = name

    return id_mapping


def split_main_data(data_path:str, output_folder:str, channel_count:int = 14):
    """
    Split the main .txt file into one csv per event.
    @param data_path: The path of the main txt file.
    @parem output_folder: Where data should be written.
    """
    databuilders = {}
    #id event device channel label size comma seperated data
    with open(data_path) as file:
        for line in file:
            
            t = line.split()
            event = t[1]
            channel = t[3]
            label = t[4]
            data = t[6].split(',')

            if(event not in databuilders):
                builder = DataBuilder(event=event, label=label, channel_count=channel_count)
                builder.__add__(data, channel=channel)
                databuilders[event] = builder
            else:
                done = databuilders[event].__add__(data, channel=channel)
                if(done):
                    databuilders[event].__writefile__(f'{output_folder}/{label}_{event}.csv')
                    databuilders.__delitem__(event)


"""
Primary use is for the original data reading, and redistributing to one file per event.
"""
class DataBuilder:
    
    def __init__(self, event:str, label:int, channel_count:int = 14):
        self.event = event
        self.label = label
        self.channel_count = channel_count
        self.df = pd.DataFrame()
    
    def __add__(self, data:list, channel:str):
        self.df[channel] = pd.Series(data)
        if(len(self.df.columns) == self.channel_count):
            return True
        return False
    
    def __writefile__(self, path: str):
        self.df.to_csv(path)