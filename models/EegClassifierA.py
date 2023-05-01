import torch.nn as nn
import torch.nn.functional as F
import torch 
"""
This is the same architecture as models.EegAutoencoder.Encoder with an extra classification head
""" 
class Classifier(nn.Module):
  def __init__(self,classes_count = 11):
    super().__init__()

    self.conv1 = nn.Conv2d(1, 6, 5)
    self.pool = nn.MaxPool2d(2, 2)
    self.conv2 = nn.Conv2d(6, 16, 5)
    self.fc1 = nn.Linear(16 * 5 * 5, 120)
    self.fc2 = nn.Linear(120, 84)
    self.fc3 = nn.Linear(84, classes_count)

  def forward(self, x):
    x = x.view(-1, 1, 32, 32)

    x = self.pool(F.relu(self.conv1(x)))
    x = self.pool(F.relu(self.conv2(x)))
    x = torch.flatten(x, 1) # flatten all dimensions except batch
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = self.fc3(x)
    return x