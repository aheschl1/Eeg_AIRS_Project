import torch.nn as nn
import torch.nn.functional as F
import torch 


class Classifier(nn.Module):
  def __init__(self, dropout_p:float = 0.5):
    super().__init__()

    self.net = nn.Sequential(

      nn.Conv1d(4, 32, 4, padding=2), # (32, 32)
      nn.ReLU(),
      nn.MaxPool1d(2),
      nn.Dropout(p=dropout_p),

      nn.Conv1d(32, 16, 4, padding=2), # (16, 16)
      nn.ReLU(),
      nn.MaxPool1d(2),
      nn.Dropout(p=dropout_p),
      
      nn.Flatten(),
      nn.Linear(1024, 64),
      nn.ReLU(),
      #nn.Dropout(p=dropout_p), #Not included in droppout graphs that do not specify linear
      nn.Linear(64, 11),
    )
#Batch norm. Fewer conv. F1 F2 regularalization Dropout layer after max pool. 
  def forward(self, x)->torch.Tensor:
    #x = x.view(-1, 4, 256)          - already this shape
    output = self.net(x)
    return F.softmax(output, dim=1)
  
  def save(self, path='./models/saved/best.pt'):
    torch.save(self.state_dict(), path)

  


class ClassifierB(nn.Module):
  def __init__(self, dropout_p:float = 0.5):
    super().__init__()

    self.net = nn.Sequential(

      nn.Conv1d(4, 32, 4, padding=2), # (32, 32)
      nn.ReLU(),
      nn.MaxPool1d(2),
      nn.Dropout(p=dropout_p),

      nn.Conv1d(32, 64, 4, padding=2), # (16, 16)
      nn.ReLU(),
      nn.Dropout(p=dropout_p),

      nn.Conv1d(64, 32, 4, padding=2), # (16, 16)
      nn.ReLU(),
      nn.MaxPool1d(2),
      
      nn.Flatten(),
      nn.Linear(2080, 64),
      nn.ReLU(),
      #nn.Dropout(p=dropout_p), #Not included in droppout graphs that do not specify linear
      nn.Linear(64, 11),
    )
#Batch norm. Fewer conv. F1 F2 regularalization Dropout layer after max pool. 
  def forward(self, x)->torch.Tensor:
    #x = x.view(-1, 4, 256)          - already this shape
    output = self.net(x)
    return F.softmax(output, dim=1)
  
  def save(self, path='./models/saved/best.pt'):
    torch.save(self.state_dict(), path)

class ClassifierC(nn.Module):
  def __init__(self, dropout_p:float = 0.5):
    super().__init__()

    self.net = nn.Sequential(

      nn.Conv1d(4, 32, 4, padding=2, dilation = 1), # (32, 32)
      nn.ReLU(),
      nn.Dropout(p=dropout_p),

      nn.Conv1d(32, 64, 4, padding=2), # (32, 32)
      nn.ReLU(),
      nn.Dropout(p=dropout_p),

      nn.MaxPool1d(6),

      nn.Flatten(),
      nn.Linear(2752, 2048),
      nn.LeakyReLU(),

      nn.Linear(2048, 1024),
      nn.LeakyReLU(),

      nn.Linear(1024, 64),
      nn.LeakyReLU(),
      #nn.Dropout(p=dropout_p), #Not included in droppout graphs that do not specify linear
      nn.Linear(64, 11),
    )
#Batch norm. Fewer conv. F1 F2 regularalization Dropout layer after max pool. 
  def forward(self, x)->torch.Tensor:
    #x = x.view(-1, 4, 256)          - already this shape
    output = self.net(x)
    return F.softmax(output, dim=1)
  
  def save(self, path='./models/saved/best.pt'):
    torch.save(self.state_dict(), path)

