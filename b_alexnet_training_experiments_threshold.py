from datasets import cifar_10
import torch
from train_branches import threshold_training
import numpy as np

batch_size_train = 512
batch_size_test = 512
input_resize, input_crop = 256, 224
train_loader, val_loader, test_loader = cifar_10(batch_size_train, batch_size_test, input_resize, input_crop)

saveRootPathResults = "."
saveRootPathModel = "."
model_name = "alexnet"

#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')

n_classes = 10

threshold_list = np.arange(0.5, 0.9, 0.1)
threshold_training(train_loader, val_loader, model_name, device, n_classes, saveRootPathResults, saveRootPathModel, threshold_list)