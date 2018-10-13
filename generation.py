import os
import argparse
import torch
import torchvision
from torch.autograd import Variable
from torchvision.utils import save_image
from my_network import Generator

def generate(G, z):
  generated = G.generate(z)
  generated = generated.mul(0.5).add(0.5).data.cpu()
  return generated

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--sample_size', type=int, default=64, help='sample size')
  parser.add_argument('--output', default=None, help='output image filename')
  parser.add_argument('--model', required=True, help='Generator model path')
  parser.add_argument('--latent_dim', required=True, type=int, help='dimension of latent variable')
  parser.add_argument('--use_gpu', action='store_true', help='GPU mode')
  opt = parser.parse_args()
  if opt.use_gpu:
    opt.use_gpu = torch.cuda.is_available()

  G = Generator(opt.latent_dim)
  G.load_state_dict(torch.load(opt.model))
  G.eval()
  if opt.use_gpu:
    G = G.cuda()

  z = Variable(torch.randn((opt.sample_size, opt.latent_dim)))
  if opt.use_gpu:
    z = z.cuda()

  generated = generate(G, z)
  if opt.output is None:
    save_image(generated, 'generated.png')
  else:
    save_image(generated, opt.output)

if __name__=='__main__':
  main()