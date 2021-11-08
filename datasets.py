import torchvision
from torch.utils.data.sampler import SubsetRandomSampler
import os, cv2, sys, time, math, os
from torchvision import transforms, utils, datasets
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler, WeightedRandomSampler
from torch.utils.data import random_split
from PIL import Image
import torch, random
import numpy as np
import torch.nn as nn

def cifar_10(batch_size_train, batch_size_test, input_resize, input_crop, split_rate=0.2):

  #To normalize the input images data.
  mean = [0.457342265910642, 0.4387686270106377, 0.4073427106250871]
  std =  [0.26753769276329037, 0.2638145880487105, 0.2776826934044154]

  # Note that we apply data augmentation in the training dataset.
  transformations_train = transforms.Compose([transforms.Resize(256),
                                              transforms.CenterCrop(224),
                                              transforms.RandomHorizontalFlip(p = 0.25),
                                              transforms.RandomRotation(25),
                                              transforms.ToTensor(), 
                                              transforms.Normalize(mean = mean, std = std),
                                              ])

  # Note that we do not apply data augmentation in the test dataset.
  transformations_test = transforms.Compose([transforms.Resize(256),
                                             transforms.CenterCrop(224), 
                                             transforms.ToTensor(), 
                                             transforms.Normalize(mean = mean, std = std),
                                             ])
  
  train_set = datasets.CIFAR10(root=".", train=True, download=True, transform=transformations_train)
  indices = np.arange(len(train_set))
  random.shuffle(indices)
  val_size = split_rate*len(train_set)
  train_size = int(len(indices) - val_size)

  train_idx, val_idx = indices[:train_size], indices[train_size:]

  train_data = torch.utils.data.Subset(train_set, indices=train_idx)
  val_data = torch.utils.data.Subset(train_set, indices=val_idx)

  train_loader = DataLoader(train_data, batch_size_train, shuffle=True, num_workers=2, pin_memory=True)
  val_loader = DataLoader(val_data, batch_size_train, shuffle=True, num_workers=2, pin_memory=True)

  test_set = datasets.CIFAR10(root=".", train=False, download=True, transform=transformations_test)
  test_loader = DataLoader(test_set, batch_size_test, shuffle=True, num_workers=2, pin_memory=True)

  return train_loader, val_loader, test_loader
