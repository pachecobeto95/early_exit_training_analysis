from tqdm import tqdm
import numpy as np
import pandas as pd

def train_main_model(branchynet, epoch, train_loader, device):
  avglosses = []
  avg_accuracies = []
  branchynet.training()

  for (data, target) in tqdm(train_loader):
    data, target = data.to(device), target.to(device)

    losses, accs = branchynet.train_main(data, target)
    avglosses.append(losses), avg_accuracies.append(accs)

  avg_loss = round(np.mean(avglosses, 0), 2)
  avg_accuracy = round(np.mean(avg_accuracies, 0), 4)

  result = {"train_loss": avg_loss, "train_acc": avg_accuracy}

  print("Train Loss: %s, Train Acc: %s"%(avg_loss, avg_accuracy))  
  
  return result

def valid_main_model(branchynet, epoch, val_loader, device):
  avglosses = []
  avg_accuracies = []
  branchynet.testing()


  with torch.no_grad():
    for (data, target) in tqdm(val_loader):
      data, target = data.to(device), target.to(device)

      losses, accs = branchynet.val_main(data, target)
      avglosses.append(losses), avg_accuracies.append(accs)

  avg_loss = round(np.mean(np.array(avglosses), 0), 2)
  avg_accuracy = round(np.mean(avg_accuracies, 0), 2)

  result = {"val_loss": avg_loss, "val_acc": avg_accuracy}

  print("Val Loss: %s, Val Acc: %s"%(avg_loss, avg_accuracy))  
  
  return result


def train_eval_main_dnn(branchynet, train_loader, test_loader, device, savePathResults):
  epoch = 0
  best_val_loss = np.inf
  patience = 5
  count = 0
  df = pd.DataFrame()

  while (count < patience):
    epoch+=1
    print("Epoch: %s"%(epoch))

    results = {}

    result_train = train_main_model(branchynet, epoch, train_loader, device)
    result_val = valid_main_model(branchynet, epoch, test_loader, device)
    
    results.update(result_train), results.update(result_val)  
    df = df.append(pd.Series(results), ignore_index=True)
    df.to_csv(savePathResults)

    if (results["val_loss"] < best_val_loss):
      best_val_loss = results["val_loss"]
      count = 0
      save_dict = {"model_state_dict": branchynet.network.state_dict(), "epoch_main": epoch, "val_main_loss": results["val_loss"],
                        "val_main_acc": results["val_acc"]}

      torch.save(save_dict, "main_dnn.pth")    
    else:
      count += 1
      print("Current Patience: %s"%(count))

  return save_dict