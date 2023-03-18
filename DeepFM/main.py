import os
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

from DeepFM.deepfm_model import deepfm
from dataset.eval import get_auc
from dataset.get_data import get_data_from_file

date="0517"
test_size = 0.1
# datasource = "steam"
datasource = "amazon"
datasource = "software"
datasource = "magazine"
featuremode = "normal"
# featuremode = "with-sentiment"
othermode = ""
# othermode = "_conti"  # 继续训练的标签
basedir = date + "_" + datasource + "_" + featuremode
base_model_save_path = "./model/" + basedir + "/" + str(date) + "_" + featuremode + othermode + "_" + datasource + "_deepfm_model"
have_old_model = False
if len(othermode) > 0:
  have_old_model = True
old_model_path = "./model/best_models/0513_normal_software_deepfm_model_epoch_128.pth"
lr = 1e-3  # learning rate
wd = 1e-4


if __name__ == "__main__":
  if not os.path.exists("./model/" + basedir):
    result = os.system("cd ./model/ && mkdir " + basedir)
    if result == 0:
      print("新建模型存储路径 {} 成功。".format(basedir))
    else:
      print("创建存储路径失败！")
  else:
    print("路径已存在！")

  batch_size = 256
  epochs = 1000
  seed = 127
  embedding_size = 16  # 离散型大部分是0/1变量
  # dnn_hidden_units = [256, 256, 128, 128, 64]  # 隐藏层单元数设置
  dnn_hidden_units = [256, 256, 256]
  if datasource == "steam":
    data, sparse_features, continuous_features, col_names = get_data_from_file(mode=featuremode, datasource=datasource)  # 返回pandas类型
  else:
    data, sparse_features, continuous_features, col_names, scoreset = get_data_from_file(mode=featuremode, datasource=datasource)  # 返回pandas类型
  print("读取数据集结束。")
  data[sparse_features] = data[sparse_features].fillna('-1',)
  data[continuous_features] = data[continuous_features].fillna('0',)
  target_col = ['label']
  data['label'] = data['label'].astype(float)  #  转换为实数

  feat_sizes = {}
  feat_sizes_continuous = {feat: 1 for feat in continuous_features}
  feat_sizes_sparse = {feat: len(data[feat].unique()) for feat in sparse_features}
  feat_sizes.update(feat_sizes_continuous)
  feat_sizes.update(feat_sizes_sparse)

  for feat in sparse_features:
    labelencoder = LabelEncoder()
    data[feat] = data[feat].astype("string")
    data[feat] = labelencoder.fit_transform(data[feat])

  minmaxscaler = MinMaxScaler(feature_range=(0, 1))  # 连续型特征归一化
  # print(data[continuous_features].dtypes)
  for feat in continuous_features:
    # print(data[feat])
    data[feat] = data[feat].astype("float")
  data[continuous_features] = minmaxscaler.fit_transform(data[continuous_features])

  label = data['label']
  train, test, label_train, label_test = train_test_split(data, label, test_size=test_size, random_state=127, stratify=label)
  features = sparse_features + continuous_features  # all feature
  # device = "cuda:0"
  model = deepfm(feat_sizes=feat_sizes, sparse_feature_columns=sparse_features, dense_feature_columns=continuous_features,
                 dnn_hidden_units=dnn_hidden_units, dnn_dropout=0.5, embedding_size=2, l2_reg_linear=1e-3)  # 实例化模型
  print("初始化模型结束。")

  # model = torch.load(old_model_path)  # 加载模型继续训练
  if have_old_model:
    model.load_state_dict(torch.load(old_model_path))  # 加载模型继续训练
    model.eval()
  train_label = pd.DataFrame(train['label'])
  train = train.drop(columns=['label'])
  train_tensor_data = TensorDataset(torch.from_numpy(np.array(train)), torch.from_numpy(np.array(train_label)))
  train_loader = DataLoader(train_tensor_data, batch_size=batch_size)

  test_label = pd.DataFrame(test['label'])
  test = test.drop(columns=['label'])
  test_tensor_data = TensorDataset(torch.from_numpy(np.array(test)), torch.from_numpy(np.array(test_label)))
  test_loader = DataLoader(test_tensor_data, batch_size=batch_size)

  loss_func = nn.BCELoss(reduction="mean")
  optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)  # 优化器

  best_auc = 0.0  # 存储auc表现最好的模型
  # 开始训练
  for epoch in range(epochs):
    total_loss_epoch = 0.0
    total_tmp = 0
    model.train()
    for index, (x, y) in enumerate(train_loader):
      x, y = x.float(), y.float()
      y_hat = model(x)  # 模型训练结果（tmp）
      optimizer.zero_grad()
      loss = loss_func(y_hat, y)
      loss.backward()
      optimizer.step()
      total_loss_epoch += loss.item()
      total_tmp += 1
    auc, acc = get_auc(test_loader, model)
    print('epoch/epoches: {}/{}, train loss: {:.4f}, test auc: {:.4f}, test acc: {:.4f}'.format(epoch, epochs,
                                                                        total_loss_epoch / total_tmp, auc, acc))
    # if (epoch % 100 == 0) and (auc > best_auc):
    if auc > best_auc:
      best_auc = auc
      model_save_path = base_model_save_path + "_epoch_" + str(epoch) + ".pth"
      print("Saving model {} ......".format(model_save_path))
      torch.save(model.state_dict(), model_save_path)  # 模型存储（只保存参数）

