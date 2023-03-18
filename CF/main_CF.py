from collections import defaultdict
from math import sqrt
from sklearn.metrics import roc_auc_score, accuracy_score, mean_squared_error, mean_absolute_error, recall_score, \
  classification_report
from sklearn.preprocessing import MinMaxScaler
from FM_solution.get_inter_data import get_data_amazon, get_data
import numpy as np

datasource = "steam"  # 数据来源
datasource = "amazon"
datasource = "software"
datasource = "magazine"

data_path = ""
if datasource == "steam":
  data_path = "../FM_solution/data/u.data"
elif datasource == "amazon":
  data_path = "../FM_solution/data/amazon_u.data"
elif datasource == "software":
  data_path = "../FM_solution/data/software_u.data"
elif datasource == "magazine":
  data_path = "../FM_solution/data/magazine_u.data"

class CF:
  def __init__(self, users, items, ratings, mode="steam", k=5, n=10):
    self.users = users  # user转换信息
    self.items = items  # item转换信息
    self.ratings = ratings
    self.k = k  # k neighborhood
    self.n = n  # 推荐个数
    self.userDict = defaultdict(list) # 用户对电影的评分列表
    self.ItemUser = defaultdict(list)  # 对电影评分的用户列表
    self.neighbors = []  # 邻居信息
    self.recommandList = []
    self.totalLabels = []   # 所有标签
    self.totalPredicts = []  # 所有预测结果
    self.cost = 0.0
    # mode决定最小得分和最大得分
    if mode == "steam":
      self.minScore = 0.0
      self.maxScore = 1.0
    else:
      self.minScore = 1.0
      self.maxScore = 5.0
    self.formatRate()

  def formatRate(self):  # 将ratings转化为userDict和ItemUser
    self.userDict = defaultdict(list) # 用户对item的评分列表
    self.ItemUser = defaultdict(list)  # 对item评分的用户列表
    for item in self.ratings:
      # print(item)
      temp = (item[1], float(item[2]))
      self.userDict[item[0]].append(temp)
      self.ItemUser[item[1]].append(item[0])

  def formatuserDict(self, usera, userb):  # 获得usera，userb评价的并集
    result = {}
    for i in self.userDict[usera]:
      result[i[0]] = [i[1], 0]
    for j in self.userDict[userb]:
      if j[0] not in result:
        result[j[0]] = [0, j[1]]
      else:
        result[j[0]][1] = j[1]
    return result

  def getCost(self, usera, userb):  # 计算余弦距离
    same_items = self.formatuserDict(usera, userb)
    x, y, z = 0.0, 0.0, 0.0
    for k, v in same_items.items():
      x += float(v[0]) * float(v[0])
      y += float(v[1]) * float(v[1])
      z += float(v[0]) * float(v[1])
    if z == 0:
      return 0
    return z / sqrt(x * y)

  def getNearestNeighbor(self, userId):  # 找到某用户的近邻用户集
    neighbors = []
    self.neighbors = []  # 邻居信息
    for i in self.userDict[userId]:
      for j in self.ItemUser[i[0]]:  # 同样产生过评分的用户
        if j != userId and j not in neighbors:
          neighbors.append(j)
    for i in neighbors:
      dist = self.getCost(userId, i)
      self.neighbors.append([dist, i])
    # dist_sum = sum(self.neighbors[0])
    dist_sum = 0.0
    for dist, i in self.neighbors:
      dist_sum += dist
    self.neighbors = [[self.modifyValue(dist, dist_sum), i] for dist, i in self.neighbors]
    self.neighbors.sort(reverse=True)  # 按距离降序
    self.neighbors = self.neighbors[:self.k]

  def modifyValue(self, value, total):  # 调整value（归一化）
    if total == 0.0:
      return 0.0
    return float(value / total)

  def modifyScore(self, score, minv, maxv):  # 调整score
    return self.minScore + (self.maxScore - self.minScore) * (score - minv) / (maxv + 0.0001 - minv)

  def getrecommendList(self, userId):  # 获取推荐列表
    self.recommandList = []
    recommandDict = defaultdict(float)
    for neighbor in self.neighbors:
      items = self.userDict[neighbor[1]]
      for item in items:
        recommandDict[item[0]] += neighbor[0] * item[1]  # 相似度*打分
    scorelist = []  # 存储打分结果，便于标准化
    for key in recommandDict:
      scorelist.append(recommandDict[key])
    min_score = min(scorelist)
    max_score = max(scorelist)

    for key in recommandDict:
      self.recommandList.append([self.modifyScore(recommandDict[key], min_score, max_score), key])
    self.recommandList.sort(reverse=True)
    self.recommandList = self.recommandList[:self.n]

  def getPrecision(self, userId):  # 计算推荐准确率（聚合后计算）
    items = [i for i in self.userDict[userId]]
    # self.totalLabels += labels
    for item, label in items:
      flag = False  # 没有预测评分
      for pred_label, recommend in self.recommandList:
        if item == recommend:
          self.totalLabels.append(label)
          self.totalPredicts.append(pred_label)
          flag = True  # 有预测评分
          break
      if not flag:
        self.totalLabels.append(label)
        self.totalPredicts.append(0)

  def recommendByUser(self, userId):
    # self.formatRate()
    # self.n = len(self.userDict[userId])
    self.n = len(self.items)  # 便于计算MAE、RMSE
    self.getNearestNeighbor(userId)
    if len(self.neighbors) > 0:
      self.getrecommendList(userId)
    # print("用户{}的推荐列表是：".format(userId))
    # print(self.recommandList)
    self.getPrecision(userId)


if __name__ == "__main__":
  data, u_trans_dic, i_trans_dic = get_data(data_path)
  u_id2name_dic = defaultdict(lambda: "")
  i_id2name_dic = defaultdict(lambda: "")
  for key in u_trans_dic:
    u_id2name_dic[u_trans_dic[key]] = key
  for key in i_trans_dic:
    i_id2name_dic[i_trans_dic[key]] = key
  data_group = []  # 三元组的形式存储打分
  for i in range(len(data)):
    data_group.append((data.loc[i, 0], data.loc[i, 1], data.loc[i, 2]))
  CF = CF(u_id2name_dic, i_id2name_dic, data_group, mode=datasource)
  for key in u_id2name_dic:
    CF.recommendByUser(key)
  labels = np.array(CF.totalLabels)
  predicts = np.array(CF.totalPredicts)
  # auc = roc_auc_score(labels, predicts)
  # acc = accuracy_score(labels, predicts)
  # print("acc: {}".format(acc))
  rmse = sqrt(mean_squared_error(labels, predicts))
  mae = mean_absolute_error(labels, predicts)
  print("mae: {:.4f}, rmse: {:.4f}.".format(mae, rmse))

  # 转换为0/1分类，计算auc，accuracy，recall
  if datasource == "steam":
    threshold = 0.5  # label为0-1时（steam）
  else:
    threshold = 3.0  # 打分区间在 1-5时（amazon）

  labels_01 = []
  predicts_01 = []
  for label in labels:
    if label > threshold:
      labels_01.append(1)
    else:
      labels_01.append(0)
  for predict in predicts:
    if predict > threshold:
      predicts_01.append(1)
    else:
      predicts_01.append(0)
  predict_amazon = []  # amazon数据计算auc用
  for predict in predicts:
    predict_amazon.append((predict - 1.0) / 4.0)
  print("预测记录数共{}条。".format(len(labels_01)))
  # print(len(predicts_01))
  if datasource == "steam":
    auc = roc_auc_score(labels_01, predicts)
  else:
    auc = roc_auc_score(labels_01, predict_amazon)
  acc = accuracy_score(labels_01, predicts_01)
  recall = recall_score(labels_01, predicts_01, average="macro")
  print("auc: {:.4f}, acc: {:.4f}, recall: {:.4f}.".format(auc, acc, recall))

  if datasource == "steam":
    print(classification_report(labels_01, predicts_01))