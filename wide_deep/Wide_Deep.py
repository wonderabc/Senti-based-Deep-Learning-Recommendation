import os
from collections import defaultdict
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from torch import nn
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
from dataset.eval import get_auc
from dataset.get_data import get_data_from_file

date = "0518"
datasource = "steam"
datasource = "amazon"
datasource = "software"
# datasource = "magazine"
test_size = 0.1

featuremode = "normal"
# featuremode = "with-sentiment"

othermode = ""
othermode = "_conti"  # 继续训练的标签

basedir = date + "_" + datasource + "_" + featuremode
base_model_save_path = "./model/" + basedir + "/" + str(date) + "_" + featuremode + othermode + "_" + datasource + "_wide_deep_model"
have_old_model = False
if len(othermode) > 0:
  have_old_model = True
old_model_path = "./model/best_models/0507_normal_software_wide_deep_model_epoch_1061.pth"
lr = 1e-4
wd = 1e-3  # 正则化参数

class DNN(nn.Module):  # DNN network
  def __init__(self, inputs_dim, hidden_units, dropout_rate,):
    super(DNN, self).__init__()
    self.inputs_dim = inputs_dim
    self.hidden_units = hidden_units
    self.dropout = nn.Dropout(dropout_rate)  # dropout层
    self.hidden_units = [inputs_dim] + list(self.hidden_units)
    self.linear = nn.ModuleList([
      nn.Linear(self.hidden_units[i], self.hidden_units[i+1]) for i in range(len(self.hidden_units) - 1)  # linear层
    ])
    for name, tensor in self.linear.named_parameters():
      if 'weight' in name:
        nn.init.normal_(tensor, mean=0, std=0.0001)  # 从给定均值和标准差的正态分布N(mean, std)中生成值，填充输入的张量或变量
    self.activation = nn.ReLU()

  def forward(self, X):
    inputs = X
    for i in range(len(self.linear)):
      fc = self.linear[i](inputs)
      fc = self.activation(fc)
      fc = self.dropout(fc)
      inputs = fc
    return inputs


class Wide_Deep(nn.Module):  # 搭建模型
  def __init__(self, feat_size, embedding_size, linear_feature_columns, dnn_feature_columns,
               user_attention=True, attention_factor=8, l2_reg=0.00001, drop_rate=0.9, dnn_hidden_units=(256, 128)):
    super(Wide_Deep, self).__init__()
    self.sparse_feature_columns = list(filter(lambda x: x[1] == 'sparse', dnn_feature_columns))  # 稀疏型特征
    self.embedding_dic = nn.ModuleDict({
      feat[0]: nn.Embedding(feat_size[feat[0]], embedding_size, sparse=False) for feat in self.sparse_feature_columns
    })
    self.continuous_feature_columns = list(filter(lambda x: x[1] == 'continuous', dnn_feature_columns))  # 连续型特征

    self.feature_index = defaultdict(int)  # feature to feature_index
    start = 0
    for feat in feat_size:
      self.feature_index[feat] = start
      start += 1

    self.dnn = DNN(len(self.continuous_feature_columns) + embedding_size * len(self.embedding_dic), dnn_hidden_units,
                   0.5)  # dnn层
    self.dnn_linear = nn.Linear(dnn_hidden_units[-1], 1, bias=False)  # (in_features, out_features, bias)
    # in_features 每个输入样本的大小；out_features  每个输出样本的大小

    # 重置dnn_hidden_units
    dnn_hidden_units = [len(feat_size), 1]
    self.linear = nn.ModuleList([
      nn.Linear(dnn_hidden_units[i], dnn_hidden_units[i+1]) for i in range(len(dnn_hidden_units) - 1)
    ])
    for name, tensor in self.linear.named_parameters():
      if 'weight' in name:
        nn.init.normal_(tensor, mean=0, std=0.00001)

    self.out = nn.Sigmoid()
    self.act = nn.ReLU()
    self.dropout = nn.Dropout(0.5)

  def forward(self, X):
    # wide
    logit = X
    for i in range(len(self.linear)):
      fc = self.linear[i](logit)
      fc = self.act(fc)
      fc = self.dropout(fc)
      logit = fc

    # deep
    for feat in self.sparse_feature_columns:
      # try:
      result = self.embedding_dic[feat[0]](X[:, self.feature_index[feat[0]]].long()).reshape(X.shape[0], 1, -1)
      # except:
        # print(X[:, self.feature_index[feat[0]]].long())
        # print(max(X[:, self.feature_index[feat[0]]].long()))
        # print(self.embedding_dic[feat[0]])

    sparse_embedding = [
      self.embedding_dic[feat[0]](X[:, self.feature_index[feat[0]]].long()).reshape(X.shape[0], 1, -1)
      for feat in self.sparse_feature_columns
    ]  # 稀疏特征embedding结果
    # print(sparse_embedding)
    sparse_input = torch.cat(sparse_embedding, dim=1)
    sparse_input = torch.flatten(sparse_input, start_dim=1)

    continuous_values = [X[:, self.feature_index[feat[0]]].reshape(-1, 1) for feat in self.continuous_feature_columns]
    continuous_input = torch.cat(continuous_values, dim=1)

    dnn_input = torch.cat((sparse_input, continuous_input), dim=1)
    dnn_out = self.dnn(dnn_input)
    dnn_logit = self.dnn_linear(dnn_out)
    logit += dnn_logit  # 加上wide部分的output

    y_pred = torch.sigmoid(logit)
    return y_pred


if __name__ == "__main__":
  if not os.path.exists("./model/" + basedir):
    result = os.system("cd ./model/ && mkdir " + basedir)
    if result == 0:
      print("新建模型存储路径 {} 成功。".format(basedir))
    else:
      print("创建存储路径失败！")
  else:
    print("路径已存在！")

  # batch_size = 1024
  batch_size = 256
  epoches = 2000
  seed = 127
  embedding_size = 1
  # dnn_hidden_units = [256, 256, 128, 128, 64]  # 隐藏层单元数设置
  dnn_hidden_units = [256, 128]
  # device = 'cuda:0'

  # sparse_features = ['C' + str(i) for i in range(1, 27)]
  # continuous_features = ['I' + str(i) for i in range(1, 14)]
  # col_names = ['label'] + continuous_features + sparse_features
  # data = pd.read_csv('./data/dac_sample.txt', names=col_names, sep='\t')
  # data = pd.read_csv('./data/criteo_sample.txt', sep=",")
  # print(data)
  data = sparse_features = continuous_features = col_names = scoreset = None
  if datasource == "steam":
    data, sparse_features, continuous_features, col_names = get_data_from_file(mode=featuremode, datasource=datasource)
  else:  # amazon / software / magazine
    data, sparse_features, continuous_features, col_names, scoreset = get_data_from_file(mode=featuremode, datasource=datasource)
    if datasource != "amazon":
      embedding_size = 16
  print("共有{}条记录，包含{}个稀疏特征，{}个连续特征。".format(len(data), len(sparse_features), len(continuous_features)))
  data[sparse_features] = data[sparse_features].fillna('-1',)
  data[continuous_features] = data[continuous_features].fillna('0',)

  target = ['label']
  feat_sizes = {}  # size of 各个特征
  feat_sizes_continuous = {feat: 1 for feat in continuous_features}
  feat_sizes_sparse = {feat: len(data[feat].unique()) for feat in sparse_features}
  # 加入到feat_sizes dic中
  feat_sizes.update(feat_sizes_continuous)
  feat_sizes.update(feat_sizes_sparse)
  # print(feat_sizes)
  data["label"] = data["label"].astype("float")
  for feat in sparse_features:
    labelencoder = LabelEncoder()
    data[feat] = data[feat].astype("string")
    data[feat] = labelencoder.fit_transform(data[feat])
  minmaxscaler = MinMaxScaler(feature_range=(0, 1))
  for feat in continuous_features:
    data[feat] = data[feat].astype("float")
  data[continuous_features] = minmaxscaler.fit_transform(data[continuous_features])
  fixlen_feature_columns = [(feat, 'sparse') for feat in sparse_features] + [(feat, 'continuous') for feat in continuous_features]
  dnn_feature_columns = fixlen_feature_columns
  linear_feature_columns = fixlen_feature_columns
  # print(dnn_feature_columns)
  label = data['label']
  train, test, label_train, label_test = train_test_split(data, label, test_size=test_size, random_state=seed, stratify=label)

  model = Wide_Deep(feat_sizes, embedding_size, linear_feature_columns, dnn_feature_columns, dnn_hidden_units=dnn_hidden_units)
  if have_old_model:
    model.load_state_dict(torch.load(old_model_path))  # 加载模型继续训练
    model.eval()

  train_label = pd.DataFrame(train['label'])
  train = train.drop(columns=['label'])
  train_tensor_data = TensorDataset(torch.from_numpy(np.array(train)), torch.from_numpy(np.array(train_label)))
  train_loader = DataLoader(train_tensor_data, shuffle=True, batch_size=batch_size)

  test_label = pd.DataFrame(test['label'])
  test = test.drop(columns=['label'])
  test_tensor_data = TensorDataset(torch.from_numpy(np.array(test)), torch.from_numpy(np.array(test_label)))
  test_loader = DataLoader(test_tensor_data, batch_size=batch_size)

  loss_func = nn.BCELoss(reduction='mean')
  optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

  best_auc = 0.0
  # 开始训练
  for epoch in range(epoches):
    total_loss_epoch = 0.0
    total_tmp = 0
    model.train()
    for index, (x, y) in enumerate(train_loader):
      x, y = x.float(), y.float()
      y_hat = model(x)
      optimizer.zero_grad()
      loss = loss_func(y_hat, y)
      loss.backward()
      optimizer.step()
      total_loss_epoch += loss.item()
      total_tmp += 1
    auc, acc = get_auc(test_loader, model)
    print("epoch/epoches: {}/{}, train loss: {:.4f}, test auc: {:.4f}, test acc: {:.4f}".
          format(epoch, epoches, total_loss_epoch / total_tmp, auc, acc))
    # if (epoch % 100 == 0) and (auc > best_auc):
    if auc > best_auc:
      model_save_path = base_model_save_path + "_epoch_" + str(epoch) + ".pth"
      best_auc = auc
      torch.save(model.state_dict(), model_save_path)  # 模型存储（只保存参数）
      print("Saving model {} ……".format(model_save_path))

