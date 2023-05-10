import numpy as np
import random
import pandas as pd

class GeneratedSample:
    
    def __init__(self, length:int = 256, freq:int=1)->None:
        self.x = np.linspace(0,np.pi, length)
        self.freq_mult = freq
        self.length = length

    def get_sample(self, sigma_min:int = 0.1, sigma_max:int = 1)->np.array:
        data = []
        for i in range(1, 5):
            channel = (i*0.5)*np.sin(self.freq_mult*self.x)
            sigma = random.uniform(sigma_min, sigma_max)
            data.append(channel*np.random.normal(1, sigma, self.length))

        data = np.array(data)
        return data, self.x
        

if __name__ == "__main__":
    test = GeneratedSample()
    test.get_sample()