import torch.nn as nn
import torch.nn.functional as F
import torch 
"""
This is the same architecture as models.EegAutoencoder.Encoder with an extra classification head
""" 
class Classifier(nn.Module):
  def __init__(self):
    super().__init__()

    self.net = nn.Sequential(
      nn.Conv1d(4, 128, 4, padding=2), # (32, 32)
      nn.ReLU(),
      nn.MaxPool1d(2),
      nn.Conv1d(128, 64, 4, padding=2), 
      nn.ReLU(),
      nn.MaxPool1d(2),
      nn.Conv1d(64, 32, 4, padding=2), # (16, 16)
      nn.ReLU(),
      nn.MaxPool1d(2),
      nn.Flatten(),
      nn.Linear(32*32, 64),
      nn.ReLU(),
      nn.Linear(64, 11),
    )
#Batch norm. Fewer conv. F1 F2 regularalizationDropout layer after max pool. 
  def forward(self, x):
    x = x.view(-1, 4, 256)
    output = self.net(x)
    return F.softmax(output, dim=1)