import os, cv2, sys, time, math, os
import torch, random
import numpy as np
import torch.nn as nn
import torchvision.models as models
from torch import Tensor
from typing import Callable, Any, Optional, List, Type, Union
import torch.nn.init as init
from branchynet import BranchyNet
from utils import norm, conv, cap

class B_AlexNet(nn.Module):
  def __init__(self, branch1, branch2, n_classes, pretrained=True):
    super(B_AlexNet, self).__init__()
    self.stages = nn.ModuleList()
    self.layers = nn.ModuleList()
    self.exits = nn.ModuleList()
    insert_branches = [2]
    self.stage_id = 0


    backbone_model = models.alexnet(pretrained=pretrained)
    backbone_model_features = backbone_model.features

    for i, layer in enumerate(backbone_model_features):
      self.layers.append(layer)
      if (i == insert_branches[0]):
        self.add_exit_point(branch1)
    
    self.layers.append(nn.AdaptiveAvgPool2d(output_size=(6, 6)))    
    self.stages.append(nn.Sequential(*self.layers))
    del self.layers   
    self.classifier = backbone_model.classifier
    self.classifier[1] = nn.Linear(9216, 4096)
    self.classifier[4] = nn.Linear(4096, 1024)
    self.classifier[6] = nn.Linear(1024, n_classes)    
    self.softmax = nn.Softmax(dim=1)

  def add_exit_point(self, branch1):
    self.stages.append(nn.Sequential(*self.layers))
    self.exits.append(nn.Sequential(*branch1))
    self.layers = nn.ModuleList()
    self.stage_id += 1    

  def forwardMain(self, x):
    for i, stage in enumerate(self.exits):
      x = self.stages[i](x)

    x = self.stages[-1](x)
    
    x = torch.flatten(x, 1)

    output = self.classifier(x)
    _, infered_class = torch.max(self.softmax(output), 1)
    return output, infered_class

  def forwardBranchesTrain(self, x, i):
    n_exits = len(self.exits)

    if(i < n_exits):
      intermediate_model = nn.Sequential(*self.stages[:(i+1)])
      x = intermediate_model(x)
      output_branch = self.exits[i](x)
      _, infered_class = torch.max(self.softmax(output_branch), 1)
      return output_branch, infered_class

    else:
      intermediate_model = nn.Sequential(*self.stages[:(i+1)])
      x = intermediate_model(x)
      x = torch.flatten(x, 1)
      output = self.classifier(x)
      _, infered_class = torch.max(self.softmax(output), 1)
      return output, infered_class


def build_b_alexnet(device, n_classes):
  pretrained = True
  branch1 = norm() + conv(64) + conv(32) + cap(512)
  b_alexnet = B_AlexNet(branch1, None, n_classes, pretrained)
  branchynet = BranchyNet(b_alexnet, device)
  return branchynet


