"""
VAE-GAN
  This implementation is based on the original implementation (https://github.com/andersbll/autoencoding_beyond_pixels).
  For details, please refer to the original paper (https://arxiv.org/pdf/1512.09300.pdf).
"""

import torch
from torch import nn

def initialize_weights(model):
  for m in model.modules():
    if isinstance(m, nn.Conv2d):
      m.weight.data.normal_(0, 0.02)
      if m.bias is not None:
        m.bias.data.zero_()
    elif isinstance(m, nn.ConvTranspose2d):
      m.weight.data.normal_(0, 0.02)
      if m.bias is not None:
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
      m.weight.data.normal_(0, 0.02)
      if m.bias is not None:
        m.bias.data.zero_()

class Encoder(nn.Module):
  """
  Encoder class

  Model:
    conv1: (nc, 64, 64) -> (64, 32, 32)
    conv2: (64, 32, 32) -> (128, 16, 16)
    conv3: (128, 16, 16) -> (256, 8, 8)
    fc4: (256*8*8) -> (2048)
    mu: (2048) -> (nz)
    logvar: (2048) -> (nz)
  """
  def __init__(self, nz):
    super(Encoder, self).__init__()

    self.conv1 = nn.Sequential(
      nn.Conv2d(3, 64, kernel_size=5, stride=2, padding=2, bias=False),
      nn.BatchNorm2d(64),
      nn.ReLU(inplace=True)
    )

    self.conv2 = nn.Sequential(
      nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2, bias=False),
      nn.BatchNorm2d(128),
      nn.ReLU(inplace=True)
    )

    self.conv3 = nn.Sequential(
      nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2, bias=False),
      nn.BatchNorm2d(256),
      nn.ReLU(inplace=True)
    )

    self.fc4 = nn.Sequential(
      nn.Linear(256 * 8 * 8, 2048, bias=False),
      nn.BatchNorm1d(2048),
      nn.ReLU(inplace=True)
    )
  
    self.mu = nn.Linear(2048, nz, bias=False)

    self.logvar = nn.Linear(2048, nz, bias=False)

    initialize_weights(self)
  
  def forward(self, x):
    batch_size = x.size(0)

    x = self.conv1(x)
    x = self.conv2(x)
    x = self.conv3(x)
    x = x.view(batch_size, -1)
    x = self.fc4(x)
    mu = self.mu(x)
    logvar = self.logvar(x)
    return mu, logvar

  
class Decoder(nn.Module):
  """
  Decoder class

  Model:
    fc1: (nz) -> (256*8*8)
    deconv2: (256, 8, 8) -> (256, 16, 16)
    deconv3: (256, 16, 16) -> (128, 32, 32)
    deconv4: (128, 32, 32) -> (32, 64, 64)
    conv5: (32, 64, 64) -> (nc, 64, 64)
  """
  def __init__(self, nz):
    super(Decoder, self).__init__()

    self.fc1 = nn.Sequential(
      nn.Linear(nz, 256 * 8 * 8, bias=False),
      nn.BatchNorm1d(256 * 8 * 8),
      nn.ReLU(inplace=True)
    )

    self.deconv2 = nn.Sequential(
      nn.ConvTranspose2d(256, 256, kernel_size=5, stride=2, padding=2, output_padding=1, bias=False),
      nn.BatchNorm2d(256),
      nn.ReLU(inplace=True)
    )

    self.deconv3 = nn.Sequential(
      nn.ConvTranspose2d(256, 128, kernel_size=5, stride=2, padding=2, output_padding=1, bias=False),
      nn.BatchNorm2d(128),
      nn.ReLU(inplace=True)
    )

    self.deconv4 = nn.Sequential(
      nn.ConvTranspose2d(128, 32, kernel_size=5, stride=2, padding=2, output_padding=1, bias=False),
      nn.BatchNorm2d(32),
      nn.ReLU(inplace=True)
    )

    self.conv5 = nn.Sequential(
      nn.Conv2d(32, 3, kernel_size=5, stride=1, padding=2),
      nn.Tanh()
    )

    initialize_weights(self)

  def forward(self, x):
    x = self.fc1(x)
    x = x.view(-1, 256, 8, 8)
    x = self.deconv2(x)
    x = self.deconv3(x)
    x = self.deconv4(x)
    x = self.conv5(x)
    return x

class Generator(nn.Module):
  """
  Generator class
  """
  def __init__(self, nz):
    super(Generator, self).__init__()

    self.encoder = Encoder(nz)
    self.decoder = Decoder(nz)
  
  def reparameterize(self, mu, logvar):
    if self.training:
      std = logvar.mul(0.5).exp_()
      eps = torch.randn_like(std)
      return eps.mul(std).add_(mu)
    else:
      return mu

  def forward(self, x):
    mu, logvar = self.encoder(x)
    z = self.reparameterize(mu, logvar)
    xhat = self.decoder(z)
    return xhat, mu, logvar
  
  def generate(self, z):
    self.eval()
    samples = self.decoder(z)
    return samples

  def reconstruct(self, x):
    self.eval()
    mu, _ = self.encoder(x)
    xhat = self.decoder(mu)

    return xhat
  

class Discriminator(nn.Module):
  """
  Discriminator class

  Model:
    conv1: (nc, 64, 64) -> (32, 64, 64)
    conv2: (32, 64, 64) -> (128, 32, 32)
    conv3: (128, 32, 32) -> (256, 16, 16)
    conv4: (256, 16, 16) -> (256, 8, 8)
    fc5: (256*8*8) -> (512)
    fc6: (512) -> (1)
  """
  def __init__(self):
    super(Discriminator, self).__init__()

    self.conv1 = nn.Sequential(
      nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=2),
      nn.ReLU(inplace=True)
    )

    self.conv2 = nn.Sequential(
      nn.Conv2d(32, 128, kernel_size=5, stride=2, padding=2, bias=False),
      nn.BatchNorm2d(128),
      nn.ReLU(inplace=True)
    )

    self.conv3 = nn.Sequential(
      nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2, bias=False),
      nn.BatchNorm2d(256),
      nn.ReLU(inplace=True)
    )

    self.conv4 = nn.Sequential(
      nn.Conv2d(256, 256, kernel_size=5, stride=2, padding=2, bias=False),
      nn.BatchNorm2d(256),
      nn.ReLU(inplace=True)
    )

    self.fc5 = nn.Sequential(
      nn.Linear(256 * 8 * 8, 512, bias=False),
      nn.BatchNorm1d(512),
      nn.ReLU(inplace=True)
    )

    self.fc6 = nn.Sequential(
      nn.Linear(512, 1),
    )

    initialize_weights(self)

  def forward(self, x):
    x = self.conv1(x)
    x = self.conv2(x)
    x = self.conv3(x)
    x = self.conv4(x)
    x = x.view(-1, 256 * 8 * 8)
    x = self.fc5(x)
    x = self.fc6(x)
    return x
