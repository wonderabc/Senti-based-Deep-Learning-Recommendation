from collections import defaultdict
from math import sqrt

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error, roc_auc_score, accuracy_score, recall_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset
from FM_solution.get_inter_data import get_data, get_data_amazon
date = "0517"
datasource = "steam"  # 选择数据源
datasource = "software"
datasource = "magazine"

data_path = ""
if datasource == "steam":
  data_path = "./data/u.data"
elif datasource == "amazon":
  data_path = "./data/amazon_u.data"
elif datasource == "software":
  data_path = "./data/software_u.data"
elif datasource == "magazine":
  data_path = "./data/magazine_u.data"

# 训练配置
epochs = 1000
batch_size = 256
seed = 127
id_embedding_dim = 256
learning_rate = 1e-3
weight_decay = 1e-4
k_dim = 10
if datasource == "steam":
  min_val, max_val = 0.0, 1.0  # steam数据 label 0-1
else:  # amazon / software / magazine
  min_val, max_val = 1.0, 5.0  # amazon数据 overall 1-5


class FmDataset(Dataset):
  def __init__(self, uid, iid, rating):
    self.uid = uid
    self.iid = iid
    self.rating = rating

  def __getitem__(self, idx):
    return self.uid[idx], self.iid[idx], self.rating[idx]

  def __len__(self):
    return len(self.uid)


class FmLayer(nn.Module):
  def __init__(self, p, k):
    super(FmLayer, self).__init__()
    self.p, self.k = p, k
    self.linear = nn.Linear(self.p, 1, bias=True)
    self.v = nn.Parameter(torch.Tensor(self.p, self.k), requires_grad=True)
    self.v.data.uniform_(-0.01, 0.01)
    self.drop = nn.Dropout(0.2)

  def forward(self, x):
    linear_out = self.linear(x)
    inter_out1 = torch.pow(torch.mm(x, self.v), 2)
    inter_out2 = torch.mm(torch.pow(x, 2), torch.pow(self.v, 2))
    pair_interactions = torch.sum(torch.sub(inter_out1, inter_out2), dim=1)
    self.drop(pair_interactions)
    output = linear_out.transpose(1, 0) + 0.5 * pair_interactions
    return output.view(-1, 1)


def train_iter(model, optimizer, data_loader, criterion):
  model.train()
  total_loss = 0
  total_len = 0
  for index, (x_u, x_i, y) in enumerate(data_loader):
    y = (y - min_val) / (max_val - min_val) + 0.01  # 归一化label
    y_pre = model(x_u, x_i)
    loss = criterion(y.view(-1, 1), y_pre)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    total_loss += loss.item() * len(y_pre)
    total_len += len(y_pre)

  loss = total_loss / total_len
  return loss


def val_iter(model, data_loader, datasource="steam"):
  model.eval()
  labels, predicts = [], []
  with torch.no_grad():
    for x_u, x_i, y in data_loader:
      y_pre = model(x_u, x_i)
      y_pre = min_val + (y_pre - 0.01) * (max_val - min_val)  # 恢复成 min_val ~ max_val
      y_pre = torch.where(y_pre > max_val, torch.full_like(y_pre, max_val), y_pre)
      y_pre = torch.where(y_pre < min_val, torch.full_like(y_pre, min_val), y_pre)
      labels.extend(y.tolist())
      predicts.extend(y_pre.tolist())
  # mse = mean_squared_error(np.array(labels), np.array(predicts))
  rmse = sqrt(mean_squared_error(np.array(labels), np.array(predicts)))
  mae = mean_absolute_error(np.array(labels), np.array(predicts))
  auc = 0.0

  if datasource == "steam":
    threshold = 0.5
  else:
    threshold = 3.0

  predicts_amazon = []  # 转为0-1以计算auc
  for predict in list(np.array(predicts)):
    predicts_amazon.append((predict - 1) / 4.0)

  predicts_tmp, labels_tmp = [], []
  for predict in list(np.array(predicts)):
    if predict > threshold:
      predicts_tmp.append(1)
    else:
      predicts_tmp.append(0)
  for label in list(np.array(labels)):
    if label > threshold:
      labels_tmp.append(1)
    else:
      labels_tmp.append(0)

  if datasource == "steam":
    auc = roc_auc_score(np.array(labels), np.array(predicts))
    acc = accuracy_score(np.array(labels_tmp), np.array(predicts_tmp))
    recall = recall_score(np.array(labels_tmp), np.array(predicts_tmp), average="macro")
  else:
    auc = roc_auc_score(np.array(labels_tmp), np.array(predicts_amazon))
    acc = accuracy_score(np.array(labels_tmp), np.array(predicts_tmp))
    recall = recall_score(np.array(labels_tmp), np.array(predicts_tmp), average="macro")
  return rmse, mae, auc, acc, recall


class FM(nn.Module):
  def __init__(self, user_nums, item_nums, id_embedding_dim):
    super(FM, self).__init__()
    # 用户/物品embedding
    self.user_id_vec = nn.Embedding(user_nums, id_embedding_dim)
    self.item_id_vec = nn.Embedding(item_nums, id_embedding_dim)
    self.fm = FmLayer(id_embedding_dim * 2, k_dim)

  def forward(self, u_id, i_id):
    u_vec = self.user_id_vec(u_id)
    i_vec = self.item_id_vec(i_id)
    x = torch.cat((u_vec, i_vec), dim=1)
    rate = self.fm(x)
    return rate


def train_model():
  # 模型初始化
  model = FM(u_max_id, i_max_id, id_embedding_dim)
  optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
  loss_func = torch.nn.MSELoss()

  best_val_mse, best_val_epoch = 10, 0
  for epoch in range(epochs):
    loss = train_iter(model, optimizer, train_loader, loss_func)
    rmse, mae, auc, acc, recall = val_iter(model, val_loader, datasource=datasource)
    print("epoch:{}, loss:{:.5f}, rmse:{:.5f}, mae:{:.5f}, auc:{:.5f}.".format(epoch, loss, rmse, mae, auc))
    if rmse < best_val_mse:
      best_val_mse, best_val_epoch = rmse, epoch
      torch.save(model, './model/best_model_' + datasource + "_" + date)
      print("Saving model in epoch {}……".format(epoch))


def test_model():
  model = torch.load("./model/best_model_" + datasource + "_" + date)
  rmse, mae, auc, acc, recall = val_iter(model, test_loader, datasource=datasource)
  print("test mae is {:.4f}, test rmse is {:.4f}, test auc is {:.4f}, test acc is {:.4f}, test recall is {:.4f}.".format(mae, rmse, auc, acc, recall))

if __name__ == "__main__":
  data, u_trans_dic, i_trans_dic = get_data(data_path)
  u_max_id = max(data[0]) + 1
  i_max_id = max(data[1]) + 1
  print("u_max_id: {}, i_max_id: {}。".format(u_max_id, i_max_id))
  x, y = data.iloc[:, :2], data.iloc[:, 2]
  x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=seed)
  x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=seed)
  train_loader = DataLoader(FmDataset(np.array(x_train[0]), np.array(x_train[1]), np.array(y_train).astype(np.float32)),
                            batch_size=batch_size)
  val_loader = DataLoader(FmDataset(np.array(x_val[0]), np.array(x_val[1]), np.array(y_val).astype(np.float32)),
                          batch_size=batch_size)
  test_loader = DataLoader(FmDataset(np.array(x_test[0]), np.array(x_test[1]), np.array(y_test).astype(np.float32)),
                           batch_size=batch_size)
  # train_model()
  test_model()
