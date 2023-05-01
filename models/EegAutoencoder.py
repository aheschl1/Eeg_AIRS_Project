import torch.nn as nn
import torch

#  defining encoder
class Encoder(nn.Module):
  def __init__(self, in_channels=1, out_channels=16, latent_dim=300):
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
    )

  def forward(self, x):
    x = x.view(-1, 1, 32, 32)
    output = self.net(x)
    return output


#  defining decoder
class Decoder(nn.Module):
  def __init__(self, in_channels=1, out_channels=16, latent_dim=300):
    super().__init__()

    self.out_channels = out_channels

    self.linear = nn.Sequential(
        nn.Linear(latent_dim, 4*out_channels*8*8),
        nn.ReLU(),
    )

    self.conv = nn.Sequential(
        nn.ConvTranspose2d(4*out_channels, 4*out_channels, 3, padding=1), # (8, 8)
        nn.ReLU(),
        nn.ConvTranspose2d(4*out_channels, 2*out_channels, 3, padding=1, stride=2, output_padding=1), # (16, 16)
        nn.ReLU(),
        nn.ConvTranspose2d(2*out_channels, 2*out_channels, 3, padding=1),
        nn.ReLU(),
        nn.ConvTranspose2d(2*out_channels, out_channels, 3, padding=1, stride=2, output_padding=1), # (32, 32)
        nn.ReLU(),
        nn.ConvTranspose2d(out_channels, out_channels, 3, padding=1),
        nn.ReLU(),
        nn.ConvTranspose2d(out_channels, in_channels, 3, padding=1)
    )

  def forward(self, x):
    output = self.linear(x)
    output = output.view(-1, 4*self.out_channels, 8, 8)
    output = self.conv(output)
    return output.view(-1, 1, 256, 4)

#  defining autoencoder
class Autoencoder(nn.Module):
  def __init__(self, encoder, decoder, device):
    super().__init__()
    self.encoder = encoder
    self.encoder.to(device)

    self.decoder = decoder
    self.decoder.to(device)

  def forward(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded