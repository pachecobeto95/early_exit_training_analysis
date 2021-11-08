import torchvision.transforms as transforms
import torchvision
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision.utils import save_image
import os, cv2, sys, time, math, os
from torchvision import transforms, utils, datasets
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler, WeightedRandomSampler
from torch.utils.data import random_split
from PIL import Image
import torch, random
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from itertools import product
import pandas as pd
import torchvision.models as models
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim import lr_scheduler
from torchvision import datasets, transforms
from torch import Tensor
from typing import Callable, Any, Optional, List, Type, Union
import torch.nn.init as init
import functools
from tqdm import tqdm
from scipy.stats import entropy
from utils import DNN


class BranchyNet:
  def __init__(self, network, device, weight_list=None, thresholdExits=None, percentTestExits=.9, percentTrainKeeps=.5, lr_main=0.005, 
               lr_branches=1.5e-4, momentum=0.9, weight_decay=0.0001, alpha=0.001, confidence_metric="confidence", 
               opt="SGD", joint=True, verbose=False):
    
    self.network = network
    self.lr_main = lr_main
    self.lr_branches = lr_branches
    self.momentum = momentum
    self.opt = opt
    self.weight_decay = weight_decay
    self.alpha = alpha
    self.joint = joint
    self.verbose = verbose
    self.thresholdExits = thresholdExits
    self.percentTestExits = percentTestExits
    self.percentTrainKeeps = percentTrainKeeps
    self.gpu = False
    self.criterion = nn.CrossEntropyLoss()
    self.weight_list = weight_list
    self.device = device
    self.confidence_metric = confidence_metric
    steps = 10

    if (weight_list is None):
      #self.weight_list = np.ones(len(self.network.stages))
      self.weight_list = np.linspace(1, 0.3, len(self.network.stages))

    
    if (self.confidence_metric == "confidence"):
      self.confidence_metric = self.compute_confidence
      self.shouldExist = self.verify_confidence

    elif (self.confidence_metric == "entropy"):
      self.confidence_metric = self.compute_entropy
      self.shouldExist = self.verify_entropy

    else:
      raise NotImplementedError("This confidence metric is not supported.")


    if (self.opt == "Adam"):
      self.optimizer_main = optim.Adam([{"params":self.network.stages.parameters()},
                                      {"params":self.network.classifier.parameters()}], lr=self.lr_main, betas=(0.9, 0.999), eps=1e-08, 
                                       weight_decay=self.weight_decay)

    else:
      self.optimizer_main = optim.SGD([{"params":self.network.stages.parameters()},
                                       {"params":self.network.classifier.parameters()}], lr=self.lr_main, momentum=self.momentum, 
                                      weight_decay=self.weight_decay)


    self.scheduler_main = optim.lr_scheduler.CosineAnnealingLR(self.optimizer_main, steps, eta_min=0, last_epoch=-1, verbose=True)


    self.optimizer_list = []
    self.scheduler_list = []

    for i in range(len(self.network.stages)):
      if(i == len(self.network.stages)-1):
        opt_branch = optim.SGD([{"params":self.network.stages.parameters()},
                                {"params":self.network.classifier.parameters()}], lr=self.lr_branches, momentum=self.momentum, 
                              weight_decay=self.weight_decay)

      else:
        opt_branch = optim.SGD([{"params":self.network.stages[i].parameters()},
                               {"params":self.network.exits.parameters()}], lr=self.lr_branches, momentum=self.momentum, 
                              weight_decay=self.weight_decay)

      self.optimizer_list.append(opt_branch)
      scheduler_branches = optim.lr_scheduler.CosineAnnealingLR(opt_branch, steps, eta_min=0, last_epoch=-1, verbose=True)
      self.scheduler_list.append(scheduler_branches)


  def training(self):
    self.network.stages.train()
    self.network.exits.train()
    self.network.classifier.train()

  def testing(self):
    self.network.stages.eval()
    self.network.exits.eval()
    self.network.classifier.eval()

  def to_gpu(self):
    self.gpu = True
    self.network = self.network.cuda()

  def to_cpu(self):
    self.gpu = False
    self.network = self.network.to("cpu")
  
  @property
  def threshold(self):
    return self.thresholdExits
  
  @threshold.setter
  def threshold(self, t):
    self.thresholdExits = t

  def compute_entropy(self, softmax_output):
    entropy_value = np.array([entropy(output) for output in softmax_output.cpu().detach().numpy()])
    return entropy_value 

  def verify_entropy(self, entropy_value, thresholdExitsValue):
    return entropy_value <= thresholdExitsValue

  def verify_confidence(self, confidence_value, thresholdExitsValue):
    return confidence_value >= thresholdExitsValue

  def compute_confidence(self, softmax_output):
    confidence_value, _ = torch.max(softmax_output, 1)
    return confidence_value.cpu().detach().numpy()

  def train_main(self, x, t):
    self.optimizer_main.zero_grad()
    output, infered_class = self.network.forwardMain(x)
    loss = self.criterion(output, t)
    loss.backward()
    self.optimizer_main.step()

    acc = 100*infered_class.eq(t.view_as(infered_class)).sum().item()/t.size(0)    
    return loss.item(), acc

  def val_main(self, x, t):
    output, infered_class = self.network.forwardMain(x)
    loss = self.criterion(output, t)
    acc = 100*infered_class.eq(t.view_as(infered_class)).sum().item()/t.size(0)    

    return loss.item(), acc

  def val_branches(self, x, t):
    remainingXVar = x
    remainingTVar = t

    numexits, losses, acc_list, acc_branches_list = [], [], [], []
    
    n_models = len(self.network.stages)
    
    n_samples = x.data.shape[0]

    softmax = nn.Softmax(dim=1)

    for i in range(n_models):
      if (remainingXVar is None) or (remainingTVar is None):
        break

      output_branch, class_infered_branch = self.network.forwardBranchesTrain(remainingXVar, i)
      
      softmax_output = softmax(output_branch)
      
      confidence_measure = self.confidence_metric(softmax_output)
      
      idx = np.zeros(confidence_measure.shape[0],dtype=bool)

      if (i == n_models-1):
        idx = np.ones(confidence_measure.shape[0],dtype=bool)
        numexit = sum(idx)
        
      else:
        if (self.thresholdExits is not None):
          min_ent = 0
          if (isinstance(self.thresholdExits, list)):
            idx[self.shouldExist(confidence_measure, self.thresholdExits[i])] = True
            numexit = sum(idx)
          else:
            idx[self.shouldExist(confidence_measure, self.thresholdExits)] = True
            numexit = sum(idx)
        
        else:
          if (isinstance(self.percentTestExits, list)):
            numexit = int((self.percentTestExits[i])*numsamples)
          else:
            numexit = int(self.percentTestExits*confidence_measure.shape[0])

          esorted = confidence_measure.argsort()
          idx[esorted[:numexit]] = True
            
      total = confidence_measure.shape[0]
      numkeep = total-numexit
      numexits.append(numexit)

      xdata = remainingXVar.data
      tdata = remainingTVar.data

      if (numkeep > 0):
        xdata_keep = xdata[~idx]
        tdata_keep = tdata[~idx]
        remainingXVar = Variable(xdata_keep, requires_grad=False).to(self.device)
        remainingTVar = Variable(tdata_keep, requires_grad=False).to(self.device)

      else:
        remainingXVar = None
        remainingTVar = None


      if (numexit > 0):
        xdata_exit = xdata[idx]
        tdata_exit = tdata[idx]                
        
        exitXVar = Variable(xdata_exit, requires_grad=False).to(self.device)
        exitTVar = Variable(tdata_exit, requires_grad=False).to(self.device)
                

        exit_output, class_infered_branch = self.network.forwardBranchesTrain(exitXVar, i)
                
        accuracy_branch = 100*class_infered_branch.eq(exitTVar.view_as(class_infered_branch)).sum().item()/exitTVar.size(0)
        acc_branches_list.append(accuracy_branch)

        loss = self.criterion(exit_output, exitTVar)
        losses.append(loss)  

      else:
        acc_branches_list.append(0.), losses.append(0.)
                
    overall_acc = 0
    overall_loss = 0
    n_accumulated_exits = np.zeros(n_models)

    losses = [loss.item() for loss in losses]
    for i, (accuracy, loss) in enumerate(zip(acc_branches_list, losses)):
      overall_acc += accuracy*numexits[i]
      overall_loss += loss*numexits[i]
      n_accumulated_exits[i] +=numexits[i]
      
    overall_acc = overall_acc/np.sum(numexits)
    overall_loss = overall_loss/np.sum(numexits)

    cdf_exits = 100*(n_accumulated_exits/np.sum(numexits))
    pct_exit_branches = 100*(np.array(numexits)/np.sum(numexits))

    return losses, overall_loss.item(), acc_branches_list, overall_acc, pct_exit_branches, cdf_exits 
  
  
  def train_branches(self, x, t):
    remainingXVar = x
    remainingTVar = t

    numexits, losses, acc_list = [], [], []
    
    n_models = len(self.network.stages)
    n_samples = x.data.shape[0]

    softmax = nn.Softmax(dim=1)

    for i in range(n_models):
      if (remainingXVar is None) or (remainingTVar is None):
        break
      
      output_branch, class_infered_branch = self.network.forwardBranchesTrain(remainingXVar, i)
      
      loss_branch = self.criterion(output_branch, remainingTVar)
      acc_branch = 100*class_infered_branch.eq(remainingTVar.view_as(class_infered_branch)).sum().item()/remainingTVar.size(0)    

      losses.append(loss_branch)
      acc_list.append(acc_branch)

      softmax_output = softmax(output_branch)
      
      confidence_measure = self.confidence_metric(softmax_output)
      
      total = confidence_measure.shape[0]

      idx = np.zeros(total, dtype=bool)

      if (i == n_models-1):
        idx = np.ones(confidence_measure.shape[0],dtype=bool)
        numexit = sum(idx)
        
      else:
        if (self.thresholdExits is not None):
          min_ent = 0
          if (isinstance(self.thresholdExits, list)):
            idx[self.shouldExist(confidence_measure, self.thresholdExits[i])] = True
            numexit = sum(idx)
          else:
            idx[self.shouldExist(confidence_measure, self.thresholdExits)] = True
            numexit = sum(idx)
        
        else:
          if (isinstance(self.percentTrainKeeps, list)):
            numkeep = int((self.percentTrainKeeps[i])*numsamples)
            numexit = int(total - numkeep)

          else:
            numkeep = int(self.percentTrainKeeps*confidence_measure.shape[0])
          
          numexit = int(total - numkeep)
          esorted = confidence_measure.argsort()
          idx[esorted[:numexit]] = True
            
      total = confidence_measure.shape[0]
      numkeep = total-numexit
      numexits.append(numexit)

      xdata = remainingXVar.data
      tdata = remainingTVar.data

      if (numkeep > 0):
        xdata_keep = xdata[~idx]
        tdata_keep = tdata[~idx]

        remainingXVar = Variable(xdata_keep, requires_grad=False).to(self.device)
        remainingTVar = Variable(tdata_keep, requires_grad=False).to(self.device)

      else:
        remainingXVar = None
        remainingTVar = None

    for i, (weight, loss) in enumerate(zip(self.weight_list, losses)):
      loss = weight*loss
      loss.backward()
            
    self.optimizer_main.step()
    [optimizer.step() for optimizer in self.optimizer_list]

    losses = np.array([loss.item() for loss in losses])
    acc_list = np.array(acc_list)

    return losses, acc_list
