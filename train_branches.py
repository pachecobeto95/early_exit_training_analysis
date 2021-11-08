import torch, os
import tqdm
import numpy as np
import pandas as pd
from b_alexnet import build_b_alexnet
from train_backbone_main import train_eval_main_dnn

def train_threshold_branches_model(branchynet, epoch, train_loader, device, threshold):

  if (threshold is not None):
    branchynet.threshold = threshold

  train_loss_branches_list, train_overall_loss_list, train_acc_branches_list, train_overall_acc_list = [], [], [], []
  
  branchynet.training()
  n_exits = len(branchynet.network.exits) + 1

  for (data, target) in tqdm(train_loader):
    data, target = data.to(device), target.to(device)

    train_loss_branches, train_acc_branches = branchynet.train_branches(data, target)

    train_loss_branches_list.append(train_loss_branches), train_acc_branches_list.append(train_acc_branches)

  avg_train_loss_branches = np.mean(train_loss_branches_list, 0)
  avg_train_acc_branches = np.mean(train_acc_branches_list, 0)

  
  result = {}  

  for i in range(n_exits):
    result.update({"train_loss_branches_%s"%(i+1):avg_train_loss_branches[i], "train_acc_branches_%s"%(i+1):avg_train_acc_branches[i]})

  return result

def valid_threshold_branches_model(branchynet, epoch, val_loader, device, threshold):

  if (threshold is not None):
    branchynet.threshold = threshold

  branchynet.testing()

  n_exits = len(branchynet.network.exits) + 1

  loss_branches_list, overall_loss_list = [], [] 
  acc_branches_list, overall_acc_list = [], []
  exit_branches_list, cdf_exits_list  = [], []

  with torch.no_grad():
    for (data, target) in tqdm(test_loader):
      data, target = data.to(device), target.to(device)

      loss_branches, overall_loss, acc_branches, overall_branches, n_exit_branches, pct_edge_exit = branchynet.val_branches(data, target)
      
      loss_branches_list.append(loss_branches), overall_loss_list.append(overall_loss)
      acc_branches_list.append(acc_branches), overall_acc_list.append(overall_branches)
      exit_branches_list.append(n_exit_branches), cdf_exits_list.append(pct_edge_exit)
      
      
  avg_loss_branches, avg_loss_overall = np.mean(loss_branches_list, 0), np.mean(overall_loss_list)
  avg_acc_branches, avg_acc_overall = np.mean(acc_branches_list, 0), np.mean(overall_acc_list)
  avg_exit_branches, avg_cdf_exits = np.mean(exit_branches_list, 0), np.mean(cdf_exits_list, 0)

  print("Overall Loss: %s, Overall Accuracy: %s"%(avg_loss_overall, avg_acc_overall))
  results = {}
  for i in range(n_exits):
    results.update({"loss_branch_%s"%(i+1): avg_loss_branches[i],
                    "acc_branch_%s"%(i+1): avg_acc_branches[i],
                    "exit_branch_%s"%(i+1): avg_exit_branches[i]})

  return results

def train_eval_threshold_branches(branchynet, train_loader, test_loader, device, savePathResults, threshold=None):
  epoch = 0
  best_val_loss = np.inf
  patience = 5
  count = 0
  df = pd.DataFrame()

  n_exits = len(branchynet.network.exits) + 1

  while (count < patience):
    epoch+=1
    print("Epoch: %s"%(epoch))

    results = {}

    result_train = train_threshold_branches_model(branchynet, epoch, train_loader, device, threshold)
    result_val = valid_threshold_branches_model(branchynet, epoch, test_loader, device, threshold)

    results.update(result_train), results.update(result_val)

    df = df.append(pd.Series(results), ignore_index=True)
    df.to_csv(savePathResults)

    if (result["val_overall_loss"] <=  best_val_loss):
      best_val_loss = results["val_loss"]
      count = 0
      save_dict = {"model_state_dict": branchynet.network.state_dict(), "epoch_branch": epoch, "val_overall_loss": results["val_overall_loss"],
                   "val_overall_acc": results["val_overall_acc"]}
      
      for i in range(n_exits):
        save_dict.update({"val_loss_branches_%s"%(i+1): results["val_loss_branches_%s"%(i+1)], 
                          "val_acc_branches_%s"%(i+1): results["val_acc_branches_%s"%(i+1)]})

    else:
      count += 1
      print("Current Patience: %s"%(count))

  return save_dict

def threshold_training(train_loader, test_loader, model_name, device, n_classes, saveRootPathResults, saveRootPathModel, threshold_list):

  branchynet = build_b_alexnet(device, n_classes)

  if (torch.cuda.is_available()):
    branchynet.to_gpu()

  save_history_main_path = os.path.join(saveRootPathResults, "history_main_%s.csv"%(model_name))
  save_main_model_dict, trained_branchynet = train_eval_main_dnn(branchynet, train_loader, test_loader, device, save_history_main_path)


  for threshold in threshold_list:
    print("Threshold: %s"%(threshold))
    save_model_dict = {}
    
    save_history_branches_path = os.path.join(saveRootPathResults, "history_branches_%s_t_%s.csv"%(model_name, threshold))

    save_model_path = os.path.join(saveRootPathModel, "model_%s_t_%s.csv"%(model_name, threshold))

    save_branches_model_dict = train_eval_threshold_branches(trained_branchynet, train_loader, test_loader, device, 
                                                             save_history_branches_path, threshold)

    save_model_path.update(save_main_model_dict), save_model_path.update(save_branches_model_dict)

    torch.save(save_model_dict, save_model_path)
