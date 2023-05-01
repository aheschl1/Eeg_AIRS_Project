import torch.nn as nn

"""
This is the same architecture as models.EegAutoencoder.Encoder with an extra classification head
""" 
class Classifier(nn.Module):
  def __init__(self, in_channels=1, out_channels=16, latent_dim=300, classes_count = 11):
    super().__init__()

    self.net = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1), # (32, 32)
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, 3, padding=1), 
        nn.ReLU(),
        nn.Conv2d(out_channels, 2*out_channels, 3, padding=1, stride=2), # (16, 16)
        nn.ReLU(),
        nn.Conv2d(2*out_channels, 2*out_channels, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(2*out_channels, 4*out_channels, 3, padding=1, stride=2), # (8, 8)
        nn.ReLU(),
        nn.Conv2d(4*out_channels, 4*out_channels, 3, padding=1),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(4*out_channels*8*8, latent_dim),
        nn.ReLU(),
        nn.Linear(latent_dim, classes_count),
        nn.Softmax(dim=1)
    )

  def forward(self, x):
    x = x.view(-1, 1, 32, 32)
    output = self.net(x)
    return output