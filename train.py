import os
import sys
import argparse
import pickle
import matplotlib.pyplot as plt
import numpy as np

import torch
import torchvision
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image

from my_network import Generator, Discriminator
from generation import generate
from reconstruction import reconstruct

from tensorboardX import SummaryWriter

def train(D, G, Dis_optimizer, Enc_optimizer, Dec_optimizer, data_loader, writer, opt, epoch=0):
  D.train()
  G.train()

  bce_loss = nn.BCEWithLogitsLoss(reduction='elementwise_mean')
  def l1_loss(input, target):
    return torch.mean(torch.abs(input - target))
  def l2_loss(input, target):
    return torch.mean((input - target).pow(2))

  t_real = Variable(torch.ones(opt.batch_size, 1))
  t_fake = Variable(torch.zeros(opt.batch_size, 1))
  if opt.use_gpu:
    t_real = t_real.cuda()
    t_fake = t_fake.cuda()

  Dis_running_loss = 0
  Enc_running_loss = 0
  Dec_running_loss = 0
  num_itrs = len(data_loader.dataset) // opt.batch_size

  for itr, (data, _) in enumerate(data_loader):
    if data.size()[0] != opt.batch_size:
      break

    # train Discriminator ===
    Dis_optimizer.zero_grad()

    x_real = Variable(data)
    z_fake_p = Variable(torch.randn(data.shape[0], opt.nz))
    if opt.use_gpu:
      x_real = x_real.cuda()
      z_fake_p = z_fake_p.cuda()
    x_fake, mu, logvar = G(x_real)

    # L_gan ---
    y_real_loss = bce_loss(D(x_real), t_real)
    y_fake_loss = bce_loss(D(x_fake), t_fake)
    y_fake_p_loss = bce_loss(D(G.decoder(z_fake_p)), t_fake)
    L_gan_real = (y_real_loss + y_fake_loss + y_fake_p_loss) / 3.0

    # Dis_loss ---
    Dis_loss = L_gan_real
    Dis_loss.backward()
    Dis_optimizer.step()
    Dis_running_loss += Dis_loss.item()

    # train Encoder ===
    Enc_optimizer.zero_grad()

    x_real = Variable(data)
    if opt.use_gpu:
      x_real = x_real.cuda()
    x_fake, mu, logvar = G(x_real)

    # L_prior ---
    L_prior = opt.alpha * (-0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp()))

    # L_llike ---
    L_recon = opt.gamma * l1_loss(x_fake, x_real)
    L_llike = l1_loss(D.feature(x_fake), D.feature(x_real))

    # Enc_loss ---
    Enc_loss = L_prior + L_recon + L_llike
    Enc_loss.backward()
    Enc_optimizer.step()
    Enc_running_loss += Enc_loss.item()

    # train Decoder ===
    Dec_optimizer.zero_grad()

    x_real = Variable(data)
    z_fake_p = Variable(torch.randn(opt.batch_size, opt.nz))
    if opt.use_gpu:
      x_real = x_real.cuda()
      z_fake_p = z_fake_p.cuda()
    x_fake, mu, logvar = G(x_real)

    # L_gan ---
    y_real_loss = bce_loss(D(x_real), t_fake)
    y_fake_loss = bce_loss(D(x_fake), t_real)
    y_fake_p_loss = bce_loss(D(G.decoder(z_fake_p)), t_real)
    L_gan_fake = (y_real_loss + y_fake_loss + y_fake_p_loss) / 3.0

    # L_llike ---
    L_recon = opt.gamma * l1_loss(x_fake, x_real)
    L_llike = l1_loss(D.feature(x_fake), D.feature(x_real))

    # Dec_loss ---
    Dec_loss = L_recon + L_llike + L_gan_fake
    Dec_loss.backward()
    Dec_optimizer.step()
    Dec_running_loss += Dec_loss.item()

    sys.stdout.write('\r\033[Kitr [{}/{}], Dis_loss: {:.6f}, Enc_loss: {:.6f}, Dec_loss: {:.6f}'.format(itr+1, num_itrs, Dis_loss.item(), Enc_loss.item(), Dec_loss.item()))
    sys.stdout.flush()

    if (itr+1) % 10 == 0:
      G.eval()

      # generation
      z = Variable(torch.randn((8, opt.nz)))
      if opt.use_gpu:
        z = z.cuda()

      generated = generate(G, z)
      writer.add_image('Generated Image', generated)

      # reconstruction
      x = Variable(data)
      x = x[:8]
      if opt.use_gpu:
        x = x.cuda()

      x, xhat = reconstruct(G, x)
      reconstructed = torch.cat((x, xhat), dim=0)
      writer.add_image('Reconstructed Image', reconstructed)

      # loss
      writer.add_scalars(
        'Loss', 
        {
          'Discriminator': Dis_loss.item(),
          'Encoder': Enc_loss.item(),
          'Decoder': Dec_loss.item(),
          'L_gan_real': L_gan_real.item(),
          'L_gan_fake': L_gan_fake.item(),
          'L_prior': L_prior.item(),
          'L_recon': L_recon.item(),
          'L_llike': L_llike.item()
        },
        global_step = epoch * num_itrs + itr + 1
      )

      G.train()

  Dis_running_loss /= num_itrs
  Enc_running_loss /= num_itrs
  Dec_running_loss /= num_itrs

  return Dis_running_loss, Enc_running_loss, Dec_running_loss


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--log_dir', default='logs', help='log directory')
  parser.add_argument('--data_dir', default='data/celeba', help='data directory')
  parser.add_argument('--batch_size', type=int, default=64, help='batch size')
  parser.add_argument('--num_workers', type=int, default=4, help='number of workers for data laoding')
  parser.add_argument('--num_epochs', type=int, default=100, help='number of epochs')
  parser.add_argument('--checkpoint', type=int, default=10, help='checkpoint epoch')
  parser.add_argument('--lr', type=float, default=2e-4, help='learning rate')
  parser.add_argument('--weight_decay', type=float, default=1e-5, help='weight decay')
  parser.add_argument('--nz', type=int, default=100, help='dimension of latent variable')
  parser.add_argument('--alpha', type=float, default=1e-2, help='coefficient of L_prior')
  parser.add_argument('--gamma', type=float, default=5, help='coefficient of L_recon')
  parser.add_argument('--G_model', default=None, help='pretrained Generator model path')
  parser.add_argument('--D_model', default=None, help='pretrained Discriminator model path')
  parser.add_argument('--use_gpu', action='store_true', help='GPU mode')
  opt = parser.parse_args()
  if opt.use_gpu:
    opt.use_gpu = torch.cuda.is_available()

  if not os.path.exists(opt.data_dir):
    os.makedirs(opt.data_dir)

  if not os.path.exists(opt.log_dir):
    os.makedirs(opt.log_dir)

  writer = SummaryWriter(os.path.join(opt.log_dir, 'runs'))

  # =============== data preparation ================ #
  transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
  ])
  dataset = datasets.ImageFolder(opt.data_dir, transform=transform)
  data_loader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers)

  # ==================== model ====================== #
  G = Generator(opt.nz)
  D = Discriminator()
  if (opt.G_model is not None) and (opt.D_model is not None):
    G.load_state_dict(torch.load(opt.G_model))
    D.load_state_dict(torch.load(opt.D_model))
  if opt.use_gpu:
    G = G.cuda()
    D = D.cuda()
    
  # ================== optimizer ==================== #
  Enc_optimizer = optim.Adam(G.encoder.parameters(), lr=opt.lr, weight_decay=opt.weight_decay, betas=(0.5, 0.999))
  Dec_optimizer = optim.Adam(G.decoder.parameters(), lr=opt.lr, weight_decay=opt.weight_decay, betas=(0.5, 0.999))
  Dis_optimizer = optim.Adam(D.parameters(), lr=opt.lr, weight_decay=opt.weight_decay, betas=(0.5, 0.999))

  # ==================== train ====================== #
  history = {}
  history['Dis_loss'] = []
  history['Enc_loss'] = []
  history['Dec_loss'] = []

  for epoch in range(opt.num_epochs):
    Dis_loss, Enc_loss, Dec_loss = train(D, G, Dis_optimizer, Enc_optimizer, Dec_optimizer, data_loader, writer, opt, epoch)
    sys.stdout.write('\r\033[Kepoch [{}/{}], Dis_loss: {:.6f}, Enc_loss: {:.6f}, Dec_loss: {:.6f}\n'.format(epoch+1, opt.num_epochs, Dis_loss, Enc_loss, Dec_loss))
    sys.stdout.flush()

    history['Dis_loss'].append(Dis_loss)
    history['Enc_loss'].append(Enc_loss)
    history['Dec_loss'].append(Dec_loss)

    if (epoch + 1) % opt.checkpoint == 0:
      # save model
      torch.save(G.state_dict(), os.path.join(opt.log_dir, 'G_epoch{:04d}.pth'.format(epoch+1)))
      torch.save(D.state_dict(), os.path.join(opt.log_dir, 'D_epoch{:04d}.pth'.format(epoch+1)))

      G.eval()

      # generation
      z = Variable(torch.randn((10, opt.nz)))
      if opt.use_gpu:
        z = z.cuda()

      generated = generate(G, z)
      save_image(generated, os.path.join(opt.log_dir, 'generated_epoch{:04d}.png'.format(epoch+1)), nrow=10)

      # reconstruction
      for data, _ in data_loader:
        x = Variable(data)
        x = x[:10]
        if opt.use_gpu:
          x = x.cuda()

        x, xhat = reconstruct(G, x)
        reconstructed = torch.cat((x, xhat), dim=0)
        save_image(reconstructed, os.path.join(opt.log_dir, 'reconstructed_epoch{:04d}.png'.format(epoch+1)), nrow=10)
        break

  # ================== save model ==================== #
  torch.save(G.state_dict(), os.path.join(opt.log_dir, 'G_epoch{:04d}.pth'.format(opt.num_epochs)))
  torch.save(D.state_dict(), os.path.join(opt.log_dir, 'D_epoch{:04d}.pth'.format(opt.num_epochs)))

  G.eval()

  # generation
  z = Variable(torch.randn((10, opt.nz)))
  if opt.use_gpu:
    z = z.cuda()

  generated = generate(G, z)
  save_image(generated, os.path.join(opt.log_dir, 'generated_epoch{:04d}.png'.format(opt.num_epochs)), nrow=10)

  # reconstruction
  for data, _ in data_loader:
    x = Variable(data)
    x = x[:10]
    if opt.use_gpu:
      x = x.cuda()

    x, xhat = reconstruct(G, x)
    reconstructed = torch.cat((x, xhat), dim=0)
    save_image(reconstructed, os.path.join(opt.log_dir, 'reconstructed_epoch{:04d}.png'.format(opt.num_epochs)), nrow=10)
    break

  # ================== show loss ===================== #
  fig = plt.figure()
  plt.subplot(3, 1, 1)
  plt.plot(history['Dis_loss'])
  plt.ylabel('Discriminator Loss')
  plt.xlabel('Epoch')
  plt.grid()

  plt.subplot(3, 1, 2)
  plt.plot(history['Enc_loss'])
  plt.ylabel('Encoder Loss')
  plt.xlabel('Epoch')
  plt.grid()

  plt.subplot(3, 1 ,3)
  plt.plot(history['Dec_loss'])
  plt.ylabel('Decoder Loss')
  plt.xlabel('Epoch')
  plt.grid()
  plt.savefig(os.path.join(opt.log_dir, 'loss.png'))

  with open(os.path.join(opt.log_dir, 'history.pkl'), 'wb') as f:
    pickle.dump(history, f)


if __name__=='__main__':
  main()
