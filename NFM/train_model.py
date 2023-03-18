import json
import os
from collections import defaultdict, namedtuple
import demjson
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from torch import nn
import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from dataset.eval import get_auc
from dataset.get_data import get_data_from_file

date="0518"
test_size = 0.1
# datasource = "steam"
datasource = "amazon"
datasource = "software"
datasource = "magazine"

featuremode = "normal"
# featuremode = "with-sentiment"

othermode = ""
othermode = "_conti"  # 继续训练的标签
basedir = date + "_" + datasource + "_" + featuremode
base_model_save_path = "./model/" + basedir + "/" + str(date) + "_" + featuremode + othermode + "_" + datasource + "_nfm_model"
have_old_model = False
if len(othermode) > 0:
  have_old_model = True
old_model_path = "./model/best_models/0517_normal_conti_magazine_nfm_model_epoch_5.pth"
lr = 1e-4  # learning rate
wd = 1e-5


class SparseFeat(namedtuple('SparseFeat', ['name', 'vocabulary_size', 'embedding_dim', 'use_hash',
                                           'dtype', 'embedding_name', 'group_name'])):
  __slots__ = ()

  def __new__(cls, name, vocabulary_size, embedding_dim=4, use_hash=False, dtype="int32", embedding_name=None,
              group_name='default_group'):
    if embedding_name is None:
      embedding_name = name
    return super(SparseFeat, cls).__new__(cls, name, vocabulary_size, embedding_dim, use_hash, dtype, embedding_name,
                                          group_name)

  def __hash__(self):
    return self.name.__hash__()

class ContinuousFeat(namedtuple('DenseFeat', ['name', 'dimension', 'dtype'])):
  __slots__ = ()

  def __new__(cls, name, dimension=1, dtype='float32'):
    return super(ContinuousFeat, cls).__new__(cls, name, dimension, dtype)

  def __hash__(self):
    return self.name.__hash__()


def activation_layer(act_name, hidden_size=None, dice_dim=2):
  if isinstance(act_name, str):
    if act_name.lower() == "sigmoid":
      act_layer = nn.Sigmoid()
    elif act_name.lower() == "relu":
      act_layer = nn.ReLU(inplace=True)
    elif act_name.lower() == "prelu":
      act_layer = nn.PReLU()
  return act_layer


class DNN(nn.Module):
  def __init__(self, inputs_dim, hidden_units, activation="prelu", l2_reg=0, dropout_rate=0.5,
               use_bn=False, init_std=0.0001, dice_dim=2, seed=127, device='cpu'):
    super(DNN, self).__init__()
    self.dropout = nn.Dropout(dropout_rate)
    self.seed = seed
    self.l2_reg = l2_reg
    # self.use_bn = use_bn
    hidden_units = [inputs_dim] + list(hidden_units)
    self.linears = nn.ModuleList([
      nn.Linear(hidden_units[i], hidden_units[i+1]) for i in range(len(hidden_units)-1)
    ])
    # if use_bn:  # batchnorm层
      # self.bn  = nn.ModuleList([
        # nn.BatchNorm1d(hidden_units[i], hidden_units[i+1]) for i in range(len(hidden_units)-1)
      # ])
    self.activation_layer = nn.ModuleList([
      activation_layer(activation, hidden_units[i+1], dice_dim) for i in range(len(hidden_units)-1)
    ])  # 激活层

    for name, tensor in self.linears.named_parameters():
      if 'weight' in name:
        nn.init.normal_(tensor, mean=0, std=init_std)

  def forward(self, inputs):
    deep_input = inputs
    for i in range(len(self.linears)):
      fc = self.linears[i](deep_input)
      # if self.use_bn:
        # fc = self.bn[i](fc)
      fc = self.activation_layer[i](fc)
      fc = self.dropout(fc)
      deep_input = fc
    return deep_input

class BiInteractionPooling(nn.Module):  # biinteraction层
  def __init__(self):
    super(BiInteractionPooling, self).__init__()
  def forward(self, inputs):
    concated_embeds_value = inputs
    square_of_sum = torch.pow(torch.sum(concated_embeds_value, dim=1, keepdim=True), 2)
    sum_of_square = torch.sum(concated_embeds_value * concated_embeds_value, dim=1, keepdim=True)
    cross_term = 0.5 * (square_of_sum - sum_of_square)
    return cross_term

class NFM(nn.Module):
  def __init__(self, feat_sizes, embedding_size, linear_feature_columns, dnn_feature_columns, dnn_hidden_units=(128, 128),
               l2_reg_embedding=1e-5, l2_reg_linear=1e-5, l2_reg_dnn=0, init_std=0.0001, seed=127, bi_dropout=0.5,
               dnn_dropout=0.5, dnn_activation='relu', task='binary', device='cpu', gpus=None):
    super(NFM, self).__init__()
    self.continuous_features_columns = list(
      filter(lambda x: isinstance(x, ContinuousFeat), dnn_feature_columns)) if len(dnn_feature_columns) else []
    continuous_input_dim = sum(map(lambda x: x.dimension, self.continuous_features_columns))

    self.sparse_features_columns = list(
      filter(lambda x: isinstance(x, SparseFeat), dnn_feature_columns)) if len(dnn_feature_columns) else []
    self.feat_sizes = feat_sizes
    self.embedding_size = embedding_size
    self.embedding_dic = nn.ModuleDict({feat.name: nn.Embedding(self.feat_sizes[feat.name], self.embedding_size, sparse=False) for feat in self.sparse_features_columns})

    for tensor in self.embedding_dic.values():
      nn.init.normal_(tensor.weight, mean=0, std=init_std)
    self.feature_idx = defaultdict(int)
    start = 0
    for feat in self.feat_sizes:
      if feat in self.feature_idx:
        continue
      self.feature_idx[feat] = start
      start += 1

    self.dnn = DNN(continuous_input_dim+self.embedding_size, dnn_hidden_units, activation=dnn_activation,
                   l2_reg=l2_reg_dnn, dropout_rate=dnn_dropout, use_bn=False, init_std=init_std, device=device)
    self.dnn_linear = nn.Linear(dnn_hidden_units[-1], 1, bias=False)
    dnn_hidden_units = [len(self.feature_idx)] + list(dnn_hidden_units) + [1]
    self.Linears = nn.ModuleList(
      [nn.Linear(dnn_hidden_units[i], dnn_hidden_units[i+1]) for i in range(len(dnn_hidden_units) - 1)]
    )
    self.relu = nn.ReLU()
    self.bi_pooling = BiInteractionPooling()
    self.bi_dropout = bi_dropout
    if self.bi_dropout > 0:
      self.dropout = nn.Dropout(bi_dropout)
    # self.dnn_dropout = nn.Dropout(dnn_dropout)
    # self.to(device)

  def forward(self, X):
    sparse_embedding = [self.embedding_dic[feat.name](X[:, self.feature_idx[feat.name]].long()).reshape(X.shape[0], 1, -1) for feat in self.sparse_features_columns]
    continuous_values = [X[:, self.feature_idx[feat.name]].reshape(-1, 1) for feat in self.continuous_features_columns]
    continuous_input = torch.cat(continuous_values, dim=1)  # 拼接

    fm_input = torch.cat(sparse_embedding, dim=1)
    # sparse_input = torch.cat(sparse_embedding, dim=1)
    # sparse_input = torch.flatten(sparse_input, start_dim=1)
    # fm_input = torch.cat((sparse_input, continuous_input), dim=1)  # 拼接sparse_embedding和continuous_input

    bi_out = self.bi_pooling(fm_input)
    if self.bi_dropout:
      bi_out = self.dropout(bi_out)
    bi_out = torch.flatten(torch.cat([bi_out], dim=-1), start_dim=1)

    dnn_input = torch.cat((continuous_input, bi_out), dim=1)
    dnn_output = self.dnn(dnn_input)
    dnn_output = self.dnn_linear(dnn_output)
    # dnn_output = self.dnn_dropout(dnn_output)

    # linear部分
    for i in range(len(self.Linears)):
      fc = self.Linears[i](X)
      fc = self.relu(fc)
      if self.bi_dropout:
        fc = self.dropout(fc)
      X = fc

    logit = X + dnn_output
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

  batch_size = 256
  epochs = 1000
  seed = 127
  embedding_size = 1  # 离散型大部分是0/1变量
  # dnn_hidden_units = [256, 256, 128, 128, 64]  # 隐藏层单元数设置
  dnn_hidden_units = [256, 128]
  # dnn_hidden_units = [128, 128]
  # sparse_features = ['C' + str(i) for i in range(1, 27)]  # 稀疏型特征
  # continuous_features = ['I' + str(i) for i in range(1, 14)]  # 连续型特征
  if datasource == "steam":
    data, sparse_features, continuous_features, col_names = get_data_from_file(mode=featuremode, datasource=datasource)  # 返回pandas类型
  else:  # amazon / software / magazine
    data, sparse_features, continuous_features, col_names, scoreset = get_data_from_file(mode=featuremode, datasource=datasource)
    if datasource != "amazon":
      embedding_size = 16
  print("读取数据集结束。")
  # data = pd.read_csv('./data/criteo_sample.txt', names=col_names, sep=',', header=0)  # 测试模型正确性
  # print(data)
  data[sparse_features] = data[sparse_features].fillna('-1',)
  data[continuous_features] = data[continuous_features].fillna('0',)

  target_col = ['label']
  data['label'] = data['label'].astype(float)  #  转换为实数
  # print(data['label'])

  feat_sizes = {}
  feat_sizes_continuous = {feat: 1 for feat in continuous_features}
  feat_sizes_sparse = {feat: len(data[feat].unique()) for feat in sparse_features}
  # print(feat_sizes)
  feat_sizes.update(feat_sizes_continuous)
  feat_sizes.update(feat_sizes_sparse)
  # print(feat_sizes)
  # 处理稀疏型特征
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

  fixlen_feature_columns = [SparseFeat(feat, data[feat].nunique()) for feat in sparse_features] \
                            + [ContinuousFeat(feat, 1,) for feat in continuous_features]
  dnn_feature_cols = fixlen_feature_columns
  linear_feature_cols = fixlen_feature_columns
  label = data['label']
  train, test, label_train, label_test = train_test_split(data, label, test_size=test_size, random_state=127, stratify=label)
  features = sparse_features + continuous_features  # all feature
  # device = "cuda:0"
  model = NFM(feat_sizes, embedding_size, linear_feature_cols, dnn_feature_cols, dnn_hidden_units=dnn_hidden_units)  # 实例化模型
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


