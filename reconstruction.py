import os
import sys
import argparse
import torch
import torchvision
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image
import matplotlib.pyplot as plt
from my_network import Generator

def reconstruct(G, x):
  xhat = G.reconstruct(x)
  x = x.mul(0.5).add(0.5).data.cpu()
  xhat = xhat.mul(0.5).add(0.5).data.cpu()
  return x, xhat

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--sample_size', type=int, default=10, help='sample size')
  parser.add_argument('--data_dir', default='data/celeba', help='data directory')
  parser.add_argument('--output', default=None, help='output image filename')
  parser.add_argument('--model', required=True, help='Generator model path')
  parser.add_argument('--latent_dim', required=True, type=int, help='dimension of latent variable')
  parser.add_argument('--use_gpu', action='store_true', help='GPU mode')
  opt = parser.parse_args()
  if opt.use_gpu:
    opt.use_gpu = torch.cuda.is_available()

  transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
  ])
  dataset = datasets.ImageFolder(opt.data_dir, transform=transform)
  data_loader = DataLoader(dataset, batch_size=opt.sample_size, shuffle=True)

  if not os.path.exists(opt.model):
    print(opt.model + ' does not exists.')
    sys.exit()
  G = Generator(opt.latent_dim)
  G.load_state_dict(torch.load(opt.model))
  G.eval()
  if opt.use_gpu:
    G = G.cuda()

  for data, _ in data_loader:
    x = Variable(data)
    if opt.use_gpu:
      x = x.cuda()

    x, xhat = reconstruct(G, x)
    result = torch.cat((x, xhat), dim=0)

    if opt.output is None:
      save_image(result, 'reconstructed.png', nrow=opt.sample_size)
    else:
      save_image(result, opt.output, nrow=opt.sample_size)

    break

if __name__=='__main__':
  main()