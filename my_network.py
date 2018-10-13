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
  def __init__(self, nz):
    super(Encoder, self).__init__()

    # input params
    self.nz = nz

    # conv1: 3 x 64 x 64 -> 32 x 64 x 64
    self.conv1 = nn.Sequential(
      nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False),
      nn.BatchNorm2d(32),
      nn.ReLU(inplace=True)
    )

    # conv2: 32 x 64 x 64 -> 64 x 32 x 32
    self.conv2 = nn.Sequential(
      nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
      nn.BatchNorm2d(64),
      nn.ReLU(inplace=True)
    )

    # conv3: 64 x 32 x 32 -> 128 x 16 x 16
    self.conv3 = nn.Sequential(
      nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
      nn.BatchNorm2d(128),
      nn.ReLU(inplace=True)
    )

    # conv4: 128 x 16 x 16 -> 256 x 8 x 8
    self.conv4 = nn.Sequential(
      nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
      nn.BatchNorm2d(256),
      nn.ReLU(inplace=True)
    )

    # conv5: 256 x 8 x 8 -> 512 x 4 x 4
    self.conv5 = nn.Sequential(
      nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=False),
      nn.BatchNorm2d(512),
      nn.ReLU(inplace=True)
    )

    # fc6: 512 * 4 * 4 -> 1024
    self.fc6 = nn.Sequential(
      nn.Linear(512 * 4 * 4, 1024, bias=False),
      nn.BatchNorm1d(1024),
      nn.ReLU(inplace=True)
    )
  
    # mu: 1024 -> nz
    self.mu = nn.Linear(1024, nz, bias=False)

    # logvar: 1024 -> nz
    self.logvar = nn.Linear(1024, nz, bias=False)

    initialize_weights(self)
  
  def forward(self, x):
    batch_size = x.size(0)

    h = self.conv1(x)
    h = self.conv2(h)
    h = self.conv3(h)
    h = self.conv4(h)
    h = self.conv5(h)
    h = h.view(batch_size, -1)
    h = self.fc6(h)
    mu = self.mu(h)
    logvar = self.logvar(h)
    return mu, logvar

  
class Decoder(nn.Module):
  def __init__(self, nz):
    super(Decoder, self).__init__()

    # input params
    self.nz = nz

    # fc1: nz -> 512 * 4 * 4
    self.fc1 = nn.Sequential(
      nn.Linear(nz, 512 * 4 * 4, bias=False),
      nn.BatchNorm1d(512 * 4 * 4),
      nn.ReLU(inplace=True)
    )

    # deconv2: 512 x 4 x 4 -> 256 x 8 x 8
    self.deconv2 = nn.Sequential(
      nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
      nn.BatchNorm2d(256),
      nn.ReLU(inplace=True)
    )

    # deconv3: 256 x 8 x 8 -> 128 x 16 x 16
    self.deconv3 = nn.Sequential(
      nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
      nn.BatchNorm2d(128),
      nn.ReLU(inplace=True)
    )

    # deconv4: 128 x 16 x 16 -> 64 x 32 x 32
    self.deconv4 = nn.Sequential(
      nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
      nn.BatchNorm2d(64),
      nn.ReLU(inplace=True)
    )

    # deconv5: 64 x 32 x 32 -> 32 x 64 x 64
    self.deconv5 = nn.Sequential(
      nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
      nn.BatchNorm2d(32),
      nn.ReLU(inplace=True)
    )

    # conv6: 32 x 64 x 64 -> 3 x 64 x 64
    self.conv6 = nn.Sequential(
      nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1, bias=False),
      nn.Tanh()
    )

    initialize_weights(self)

  def forward(self, z):
    xhat = self.fc1(z)
    xhat = xhat.view(-1, 512, 4, 4)
    xhat = self.deconv2(xhat)
    xhat = self.deconv3(xhat)
    xhat = self.deconv4(xhat)
    xhat = self.deconv5(xhat)
    xhat = self.conv6(xhat)
    return xhat

class Generator(nn.Module):
  def __init__(self, nz):
    super(Generator, self).__init__()

    # input params
    self.nz = nz

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
  
  def encode(self, x):
    mu, logvar = self.encoder(x)
    return mu

  def generate(self, z):
    self.eval()
    samples = self.decoder(z)
    return samples

  def reconstruct(self, x):
    self.eval()
    mu, logvar = self.encoder(x)
    xhat = self.decoder(mu)

    return xhat
  
  def save_model(self, path):
    torch.save(self.state_dict(), path)

class Discriminator(nn.Module):
  def __init__(self):
    super(Discriminator, self).__init__()

    # conv1: 3 x 64 x 64 -> 32 x 64 x 64
    self.conv1 = nn.Sequential(
      nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False),
      nn.BatchNorm2d(32),
      nn.LeakyReLU(0.2)
    )

    # conv2: 32 x 64 x 64 -> 64 x 32 x 32
    self.conv2 = nn.Sequential(
      nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
      nn.BatchNorm2d(64),
      nn.LeakyReLU(0.2)
    )

    # conv3: 64 x 32 x 32 -> 128 x 16 x 16
    self.conv3 = nn.Sequential(
      nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
      nn.BatchNorm2d(128),
      nn.LeakyReLU(0.2)
    )

    # conv4: 128 x 16 x 16 -> 256 x 8 x 8
    self.conv4 = nn.Sequential(
      nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
      nn.BatchNorm2d(256),
      nn.LeakyReLU(0.2)
    )

    # conv5: 256 x 8 x 8 -> 512 x 4 x 4
    self.conv5 = nn.Sequential(
      nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=False),
      nn.BatchNorm2d(512),
      nn.LeakyReLU(0.2)
    )

    # fc6: 512 * 4 * 4 -> 1024
    self.fc6 = nn.Sequential(
      nn.Linear(512 * 4 * 4, 1024, bias=False),
      nn.BatchNorm1d(1024),
      nn.LeakyReLU(0.2)
    )

    # fc7: 1024 -> 1
    self.fc7 = nn.Sequential(
      nn.Linear(1024, 1),
    )

    initialize_weights(self)

  def forward(self, x):
    f = self.conv1(x)
    f = self.conv2(f)
    f = self.conv3(f)
    f = self.conv4(f)
    f = self.conv5(f)
    f = f.view(-1, 512 * 4 * 4)
    f = self.fc6(f)
    o = self.fc7(f)
    return o
  
  def feature(self, x):
    f = self.conv1(x)
    f = self.conv2(f)
    f = self.conv3(f)
    return f.view(-1, 128 * 16 * 16)
  
  def fc_feature(self, x):
    f = self.conv1(x)
    f = self.conv2(f)
    f = self.conv3(f)
    f = self.conv4(f)
    f = self.conv5(f)
    f = f.view(-1, 512 * 4 * 4)
    f = self.fc6(f)
    return f

  def save_model(self, path):
    torch.save(self.state_dict(), path)