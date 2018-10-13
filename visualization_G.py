import os
import sys
import shutil
import argparse
import numpy as np
from collections import OrderedDict
from sklearn import decomposition
import matplotlib.pyplot as plt
import torch
import torchvision
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from my_network import Generator

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--sample_size', type=int, default=1000, help='sample_size')
  parser.add_argument('--latent_dim', required=True, type=int, help='dimension of latent variable')
  parser.add_argument('--log_dir', required=True, help='log directory')
  parser.add_argument('--prefix', default='', help='prefix of output file')
  parser.add_argument('--suffix', default='', help='suffix of output file')
  parser.add_argument('--fit', default='all', help='which data PCA fits (real / all)')
  parser.add_argument('--num_workers', type=int, default=4, help='number of workers for data loading')
  parser.add_argument('--real_data', default='data/celeba', help='real face dataset')
  parser.add_argument('--fake_data_vj', default='data/places365_vj', help='fake-face dataset detected by vj')
  parser.add_argument('--fake_data_cnn', default='data/places365_cnn', help='fake-face dataset detected by cnn')
  parser.add_argument('--simulacra_data', default='data/simulacra_nopad', help='simulacra face dataset') 
  parser.add_argument('--negative_data', default='data/places365', help='negative face dataset')
  parser.add_argument('--model', required=True, help='Generator model path')
  parser.add_argument('--use_gpu', action='store_true', help='GPU mode')
  opt = parser.parse_args()
  if opt.use_gpu:
    opt.use_gpu = torch.cuda.is_available()

  transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
  ])

  if not os.path.exists(opt.log_dir):
    os.makedirs(opt.log_dir)

  if not os.path.exists(opt.model):
    print(opt.model + ' does not exists.')
    sys.exit()
  G = Generator(opt.latent_dim)
  G.load_state_dict(torch.load(opt.model))
  G.eval()
  if opt.use_gpu:
    G = G.cuda()

  latent = OrderedDict()

  # real face
  dataset = datasets.ImageFolder(opt.real_data, transform=transform)
  data_loader = DataLoader(dataset, batch_size=opt.sample_size, shuffle=True, num_workers=opt.num_workers)
  for data, _ in data_loader:
    x = Variable(data)
    if opt.use_gpu:
      x = x.cuda()

    real = G.encode(x).data.cpu().numpy()
    latent["real face"] = real
    break

  # fake face_vj
  if opt.fake_data_vj is not None:
    dataset = datasets.ImageFolder(opt.fake_data_vj, transform=transform)
    data_loader = DataLoader(dataset, batch_size=opt.sample_size, num_workers=opt.num_workers)
    for data, _ in data_loader:
      x = Variable(data)
      if opt.use_gpu:
        x = x.cuda()

      vj = G.encode(x).data.cpu().numpy()
      latent["fake face (vj)"] = vj
      break
    
  # fake-face cnn
  if opt.fake_data_cnn is not None:
    dataset = datasets.ImageFolder(opt.fake_data_cnn, transform=transform)
    data_loader = DataLoader(dataset, batch_size=opt.sample_size, num_workers=opt.num_workers)
    for data, _ in data_loader:
      x = Variable(data)
      if opt.use_gpu:
        x = x.cuda()
      
      cnn = G.encode(x).data.cpu().numpy()
      latent["fake face (cnn)"] = cnn
      break

  # simulacra face
  if opt.simulacra_data is not None:
    dataset = datasets.ImageFolder(opt.simulacra_data, transform=transform)
    data_loader = DataLoader(dataset, batch_size=opt.sample_size, num_workers=opt.num_workers)
    for data, _ in data_loader:
      x = Variable(data)
      if opt.use_gpu:
        x = x.cuda()
      
      simulacra = G.encode(x).data.cpu().numpy()
      latent["simulacra face"] = simulacra
      break

  # non-face
  if opt.negative_data is not None:
    dataset = datasets.ImageFolder(opt.negative_data, transform=transform)
    data_loader = DataLoader(dataset, batch_size=opt.sample_size, num_workers=opt.num_workers)
    for data, _ in data_loader:
      x = Variable(data)
      if opt.use_gpu:
        x = x.cuda()
      
      negative = G.encode(x).data.cpu().numpy()
      latent["negative face"] = negative
      break

  pca = decomposition.PCA(n_components=2)
  if opt.fit == 'real':
    pca.fit(latent['real face'])
  elif opt.fit == 'all':
    all_data = np.array(latent['real face'])
    for key in latent:
      if not key == 'real face':
        np.concatenate([all_data, latent[key]], axis=0)
    pca.fit(all_data)
  else:
    print('"fit" option is invalid.')
    sys.exit(1)

  plt.figure(figsize=(10,10))
  for key in latent:
    if key == 'real face':
      plt.scatter(pca.fit_transform(latent[key])[:,0], pca.fit_transform(latent[key])[:,1], label=key, c='r', s=5, alpha=0.7, edgecolors='none')
    elif key == 'simulacra face':
      plt.scatter(pca.fit_transform(latent[key])[:,0], pca.fit_transform(latent[key])[:,1], label=key, c='y', s=5, alpha=0.7, edgecolors='none')
    elif key == 'negative face':
      plt.scatter(pca.fit_transform(latent[key])[:,0], pca.fit_transform(latent[key])[:,1], label=key, c='k', s=5, alpha=0.7, edgecolors='none')
    elif key == 'fake face (cnn)':
      plt.scatter(pca.fit_transform(latent[key])[:,0], pca.fit_transform(latent[key])[:,1], label=key, c='b', s=5, alpha=0.7, edgecolors='none')
    elif key == 'fake face (vj)':
      plt.scatter(pca.fit_transform(latent[key])[:,0], pca.fit_transform(latent[key])[:,1], label=key, c='g', s=5, alpha=0.7, edgecolors='none')
  plt.legend()
  plt.xlim(-10, 10)
  plt.ylim(-10, 10)
  if opt.fit == 'real':
    plt.savefig(os.path.join(opt.log_dir, opt.prefix + 'fit-real' + opt.suffix + '.png'), format='png', dpi=500)
  elif opt.fit == 'all':
    plt.savefig(os.path.join(opt.log_dir, opt.prefix + 'fit-all' + opt.suffix + '.png'), format='png', dpi=500) 

if __name__=='__main__':
  main()
