import torch.nn as nn
import torch.nn.functional as F
import torch 


class Classifier(nn.Module):
  def __init__(self, dropout_p:float = 0.5, num_classes:int=10):
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
      nn.Linear(64, num_classes),
    )
#Batch norm. Fewer conv. F1 F2 regularalization Dropout layer after max pool. 
  def forward(self, x)->torch.Tensor:
    #x = x.view(-1, 4, 256)          - already this shape
    output = self.net(x)
    return F.softmax(output, dim=1)
  
  def save(self, path='./models/saved/best.pt'):
    torch.save(self.state_dict(), path)

class ClassifierImagenet(nn.Module):
  def __init__(self, dropout_p:float = 0.5, num_classes:int=10):
    super().__init__()

    self.net = nn.Sequential(

      nn.Conv1d(4, 32, 4, padding=2), # (32, 32)
      nn.ReLU(),
      nn.MaxPool1d(2),
      nn.Dropout(p=dropout_p),

      nn.Conv1d(32, 64, 4, padding=2), # (64, 128)
      nn.ReLU(),
      nn.MaxPool1d(2),                 # (64 64)
      nn.Dropout(p=dropout_p),

      nn.Conv1d(64, 16, 4, padding=2), # (16, 64)
      nn.ReLU(),
      nn.Dropout(p=dropout_p),
      
      nn.Flatten(),
      nn.Linear(1216, num_classes),
    )
#Batch norm. Fewer conv. F1 F2 regularalization Dropout layer after max pool. 
  def forward(self, x)->torch.Tensor:
    #x = x.view(-1, 4, 256)          - already this shape
    output = self.net(x)
    return F.softmax(output, dim=1)
  
  def save(self, path='./models/saved/best.pt'):
    torch.save(self.state_dict(), path)

class ClassifierBNorm(nn.Module):
  def __init__(self, dropout_p:float = 0.5):
    super().__init__()

    self.net = nn.Sequential(

      nn.Conv1d(4, 32, 4, padding=2), # (32, 32)
      nn.ReLU(),
      nn.MaxPool1d(2),
      nn.BatchNorm1d(32),
      nn.Dropout(p=dropout_p),

      nn.Conv1d(32, 16, 4, padding=2), # (16, 16)
      nn.ReLU(),
      nn.MaxPool1d(2),
      nn.Dropout(p=dropout_p),
      
      nn.Flatten(),
      nn.Linear(1024, 64),
      nn.ReLU(),
      #nn.Dropout(p=dropout_p), #Not included in droppout graphs that do not specify linear
      nn.Linear(64, 10),
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

      nn.Conv1d(64, 32, 4, padding=2), # (16, 16)
      nn.ReLU(),
      nn.MaxPool1d(2),
      nn.Dropout(p=dropout_p),
      
      nn.Flatten(),
      nn.Linear(2080, 1040),
      nn.LeakyReLU(),
      nn.Linear(1040, 400),
      nn.ReLU(),
      nn.Linear(400, 100),
      nn.ReLU(),
      nn.Linear(100, 10),
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
      nn.BatchNorm1d(32),

      nn.Conv1d(32, 64, 4, padding=2), # (32, 32)
      nn.ReLU(),
      nn.Dropout(p=dropout_p),
      nn.MaxPool1d(6),

      nn.Flatten(),
      nn.Linear(2752, 2048),
      nn.ReLU(),

      nn.Linear(2048, 1024),
      nn.ReLU(),

      nn.Linear(1024, 64),
      nn.ReLU(),
      #nn.Dropout(p=dropout_p), #Not included in droppout graphs that do not specify linear
      nn.Linear(64, 10),
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

      nn.Conv1d(64, 32, 4, padding=2), # (16, 16)
      nn.ReLU(),
      nn.MaxPool1d(2),
      nn.Dropout(p=dropout_p),
      
      nn.Flatten(),
      nn.Linear(2080, 1040),
      nn.LeakyReLU(),
      nn.Linear(1040, 400),
      nn.ReLU(),
      nn.Linear(400, 100),
      nn.ReLU(),
      nn.Linear(100, 10),
    )
#Batch norm. Fewer conv. F1 F2 regularalization Dropout layer after max pool. 
  def forward(self, x)->torch.Tensor:
    #x = x.view(-1, 4, 256)          - already this shape
    output = self.net(x)
    return F.softmax(output, dim=1)
  
  def save(self, path='./models/saved/best.pt'):
    torch.save(self.state_dict(), path)


class FullyConnected(nn.Module):
  def __init__(self, dropout_p:float = 0.5):
    super().__init__()

    self.net = nn.Sequential(
      nn.Linear(256, 64),
      nn.ReLU(),
      nn.Dropout1d(),

      nn.Linear(64, 32),
      nn.ReLU(),
      nn.Dropout1d(),

      nn.Linear(32, 16),
      nn.LeakyReLU(),
      nn.Linear(16, 10),
    )
#Batch norm. Fewer conv. F1 F2 regularalization Dropout layer after max pool. 
  def forward(self, x)->torch.Tensor:
    #x = x.view(-1, 4, 256)          - already this shape
    output = self.net(x)
    return F.softmax(output, dim=1)
  
  def save(self, path='./models/saved/best.pt'):
    torch.save(self.state_dict(), path)

class EegWithLSTM(nn.Module):
  def __init__(self, num_classes, dropout_p:float=0.5):
    super().__init__()
    self.lstm = nn.LSTM(input_size=4, hidden_size=300, num_layers=1, batch_first=True)

    self.net = nn.Sequential(
      nn.Conv1d(4, 32, 4, padding=2), # (32, 32)
      nn.ReLU(),
      nn.MaxPool1d(2),
      nn.Dropout(p=dropout_p),

      nn.Conv1d(32, 64, 4, padding=2), # (64, 128)
      nn.ReLU(),
      nn.MaxPool1d(2),                 # (64 64)
      nn.Dropout(p=dropout_p),

      nn.Conv1d(64, 16, 4, padding=2), # (16, 64)
      nn.ReLU(),
      nn.Dropout(p=dropout_p),
      
      nn.Flatten(),
      nn.Linear(1216, num_classes),
    )
  
  def forward(self, x)->torch.Tensor:
    x = x.view(-1, 300, 4)

    h_0 = torch.zeros(x.shape[0], 4, 300)
    c_0 = torch.zeros(x.shape[0], 4, 300)

    output, (final_hidden_state, final_cell_state) = self.lstm(x, (h_0, c_0))

    print(final_hidden_state.shape)

    output = self.net(final_hidden_state)
    return F.softmax(output, dim=1)
  
  def save(self, path='./models/saved/best.pt'):
    torch.save(self.state_dict(), path)
 