# 生成训练集
# 2023.1.15 新增功能，读取五分类数据集（standFord nlp 数据库，即SST5数据集）
# 2023.2.9 新增功能，处理公开数据集（Yelp--yelp / IMDB Movie--imdb / Tweet Airline--tweet）
import json
import os
from collections import defaultdict
import random
import xlrd
from keras.utils import to_categorical
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import numpy as np
random.seed(127)  # 控制随机状态

# 2023.2.9 新增
# 极性转换
def Transform5to3(label):
  if label == 0 or label == 1:
    return 0
  if label == 2:
    return 2
  if label == 3 or label == 4:
    return 1

def Transform5to2(label):
  if label == 0 or label == 1:
    return 0
  # 2的样本舍去
  if label == 3 or label == 4:
    return 1
  return -1

def Transform3to2(label):
  if label < 2:  # 2舍去
    return label
  return -1


# 划分训练集、测试集的方法①
def get_data(neg, pos, neu, train_rate=0.9):  # 划分Amazon数据集的旧方法
  random.seed(127)
  x_train, y_train = [], []
  x_test, y_test = [], []
  traindata, testdata = [], []
  traindata.extend(random.sample(neg, int(len(neg) * train_rate)))
  testdata.extend([(x, y) for x, y in neg if (x, y) not in traindata])
  traindata.extend(random.sample(pos, int(len(pos) * train_rate)))
  testdata.extend([(x, y) for x, y in pos if (x, y) not in traindata])
  traindata.extend(random.sample(neu, int(len(neu) * train_rate)))
  testdata.extend([(x, y) for x, y in neu if (x, y) not in traindata])
  # traindata = traindata[:200]  # debug用，减小训练集规模
  for x, y in traindata:
    tmp_x = x.replace(",", " ").replace(".", " ").replace(";", " ").split(" ")
    x_train.append(tmp_x)
    # y_train.append(np.argmax(y))
    y_train.append(y)

  for x, y in testdata:
    tmp_x = x.replace(",", " ").replace(".", " ").replace(";", " ").split(" ")
    x_test.append(tmp_x)
    # y_test.append(np.argmax(y))
    y_test.append(y)
  return x_train, y_train, x_test, y_test


# 划分训练集、测试集的方法②
def GetOpenData(dataset, train_rate=0.9):  # 处理公开数据集
  x_train, x_test, y_train, y_test = [], [], [], []
  for key in dataset:
    DataList = dataset[key]
    XData = [x for x, y in DataList]
    YData = [y for x, y in DataList]
    tmp_x_train, tmp_x_test, tmp_y_train, tmp_y_test = train_test_split(XData, YData, train_size=train_rate,
                                                                        random_state=127)
    x_train.extend(tmp_x_train)
    x_test.extend(tmp_x_test)
    y_train.extend(tmp_y_train)
    y_test.extend(tmp_y_test)
  return x_train, y_train, x_test, y_test


class OpenDataPath:  # 公开数据集的存储路径及类别信息
  def __init__(self, name="yelp"):
    self.datapath = ""
    if name == "yelp":
      self.datapath = "F:/【Dataset】/Yelp/yelp_academic_dataset_review.json"
      # self.CateNum = 5
    elif name == "imdb":
      self.datapath = "F:/多分类训练集/二分类/IMDB_Dataset.csv"
      # self.CateNum = 2
    elif name == "tweet":
      self.datapath = "F:/多分类训练集/三分类/Tweets.csv"
      # self.CateNum = 3

def GetYelpReviewData(datapath, scale, polarity_total, to_onehot=True):
  """
  :param to_onehot:  是否需要转化为onehot向量
  :param polarity_total: 情感分类类别数
  :param datapath: 存储路径
  :param scale: 数据集规模，保证可以整除类别数
  :return: dataset: 类别均衡的数据集，未划分训练集、验证集、测试集
  """
  # yelp 原始数据集是五分类
  data_file = open(datapath, "r", encoding="utf8").readlines()
  dataset = defaultdict(list)
  if scale == -1:
    scale = len(data_file)
  polarity_each = scale // polarity_total
  for i in tqdm(range(len(data_file))):
    line = data_file[i]
    item = json.loads(line)
    polarity = item["stars"] - 1  # 需要从1-5分映射到0-4分
    if polarity_total == 3:
      polarity = Transform5to3(polarity)
    elif polarity_total == 2:
      polarity = Transform5to2(polarity)
    if polarity == -1:  # 是被舍去的样本
      continue
    new_polarity = polarity
    if to_onehot:
      new_polarity = to_categorical(polarity, polarity_total)
    content = item["text"].strip()
    dataset[polarity].append((content, new_polarity))
  print("抽样前样本规模：")
  for key in dataset:
    print(key, len(dataset[key]))
    dataset[key] = random.sample(dataset[key], min(len(dataset[key]), polarity_each))  # 随机抽样
  return dataset

def GetIMDBReviewData(datapath, scale, polarity_total, to_onehot=True):
  data_file = pd.read_csv(datapath)
  dataset = defaultdict(list)
  if scale == -1:
    scale = len(data_file)
  polarity_each = scale // polarity_total
  for i in tqdm(range(len(data_file))):
    item = data_file.loc[i]
    polarity = item["sentiment"].strip()
    if polarity == "positive":
      polarity = 1
    else:
      polarity = 0
    new_polarity = polarity
    if to_onehot:
      new_polarity = to_categorical(polarity, polarity_total)
    content = item["review"].strip()
    dataset[polarity].append((content, new_polarity))
  print("抽样前样本规模：")
  for key in dataset:
    print(key, len(dataset[key]))
    dataset[key] = random.sample(dataset[key], min(len(dataset[key]), polarity_each))  # 随机抽样
  return dataset

def GetTweetReviewData(datapath, scale, polarity_total, to_onehot=True):
  data_file = pd.read_csv(datapath)
  dataset = defaultdict(list)
  if scale == -1:
    scale = len(data_file)
  polarity_each = scale // polarity_total
  for i in tqdm(range(len(data_file))):
    item = data_file.loc[i]
    polarity = item["airline_sentiment"].strip()
    new_polarity = polarity
    content = item["text"].strip()
    if polarity == "positive":
      polarity = 1
    elif polarity == "neutral":
      polarity = 2
    else:
      polarity = 0
    if polarity_total == 2:
      polarity = Transform3to2(polarity)
    if polarity == -1:  # 是被舍去的样本
      continue
    if to_onehot:
      new_polarity = to_categorical(polarity, polarity_total)
    dataset[polarity].append((content, new_polarity))
  print("抽样前样本规模：")
  for key in dataset:
    print(key, len(dataset[key]))
    dataset[key] = random.sample(dataset[key], min(len(dataset[key]), polarity_each))
  return dataset

class OpenData:
  def __init__(self, name="yelp", scale=10000, NoScaleLimit=False, CateNum=3):
    PATH = OpenDataPath(name)
    self.scale = scale  # 数据集规模
    if NoScaleLimit:
      self.scale = -1  # 代表没有数据集规模限制，依据读取的文件决定
    self.datapath = PATH.datapath
    self.CateNum = CateNum
    self.dataset = defaultdict(list)
    if name == "yelp":
      self.dataset = GetYelpReviewData(self.datapath, self.scale, self.CateNum)
    elif name == "imdb":
      self.dataset = GetIMDBReviewData(self.datapath, self.scale, self.CateNum)
    elif name == "tweet":
      self.dataset = GetTweetReviewData(self.datapath, self.scale, self.CateNum)
    print("抽样后样本规模：")
    for key in self.dataset:
      print(key, len(self.dataset[key]))

current_directory = "D:/Workspace/workspace/Steam_Recommend"  # 整个工程的文件目录
path = current_directory + "/data/sentiment_train_set/labeled"
# path = "mnt/data/sentiment_train_set/labeled"  # cloud version
cate5_path = current_directory + "/data/sentiment_train_set/labeled/5cate"  # 五分类数据集

# 2023.1.15新增
def GetLabeledData(file, to_onehot=True):
  f = open(file, "r", encoding="utf8")
  lines = f.readlines()
  dataset = []
  polarity_info = defaultdict(int)
  for line in lines:
    polarity, content = line.strip().split("\t")
    polarity = int(polarity[-1])
    polarity_info[polarity] += 1
    if to_onehot:
      polarity = to_categorical(polarity, 5)
    dataset.append((content, polarity))
  # print(file, "SST5数据集基本情况：")
  # for key in polarity_info:
    # print(key, polarity_info[key])
  return dataset

def get_labeled_data_5cate(to_onehot=True):
  # 五分类数据集已实现类别均衡，且已清洗去重，无需重复处理
  train_file = os.path.join(cate5_path, "sst_train.txt")
  dev_file = os.path.join(cate5_path, "sst_dev.txt")
  test_file = os.path.join(cate5_path, "sst_test.txt")
  Trainset = GetLabeledData(train_file,to_onehot)
  Devset = GetLabeledData(dev_file, to_onehot)
  Testset = GetLabeledData(test_file, to_onehot)
  Trainset = np.array(Trainset)
  Devset = np.array(Devset)
  Testset = np.array(Testset)
  return Trainset, Devset, Testset

# 旧函数
def get_labeled_data(to_onehot=True):
    filenames = os.listdir(path)
    contentlist = []
    dataset = []  # 训练集 [(content, polarity)]
    neg_cnt, neu_cnt, pos_cnt = 0, 0, 0  # 消极、中性、积极评价的条数
    maxlen = 0  # 句子的最大长度
    for file in filenames:
        # if file == "4cate" or file == "5cate":  # 下属文件夹不访问
          # continue
        filepath = path + "/" + file
        if os.path.isdir(filepath):  # 下属文件夹不访问
          continue
        data = xlrd.open_workbook(filepath)
        sheetnames = data.sheet_names()
        # print(sheetnames)
        for sn in sheetnames:
            table = data.sheet_by_name(sn)
            rowNum = table.nrows
            colNum = table.ncols
            # print(rowNum, colNum)
            for rownum in range(1, rowNum):
                content = str(table.cell_value(rownum, 1))
                if len(content) == 0:
                    continue
                contentlen = len([c for c in content.split() if len(c.strip()) > 0])
                if contentlen == 780:  # 处理重复表达
                    content = "SUPER HOT"
                    contentlen = 2
                maxlen = max(maxlen, contentlen)
                polarity = table.cell_value(rownum, 2)
                if to_onehot:
                  polarity = to_categorical(polarity, 3)  # 转化为one hot向量，类似[0 1 0]
                  if content not in contentlist:
                    contentlist.append(content)
                    dataset.append((content, polarity))
                    if polarity[0] == 1:
                        neg_cnt += 1
                    elif polarity[1] == 1:
                        pos_cnt += 1
                    else:
                        neu_cnt += 1
                else:
                  if content not in contentlist:
                    contentlist.append(content)
                    dataset.append((content, polarity))
                    if polarity == 0:
                      neg_cnt += 1
                    elif polarity == 1:
                      pos_cnt += 1
                    else:
                      neu_cnt += 1
    print(neg_cnt, pos_cnt, neu_cnt, maxlen)
    negset = [(content, polarity) for content, polarity in dataset if polarity[0] == 1]
    posset = [(content, polarity) for content, polarity in dataset if polarity[1] == 1]
    neuset = [(content, polarity) for content, polarity in dataset if polarity[2] == 1]
    return dataset, negset, posset, neuset, maxlen


if __name__ == "__main__":
  # get_labeled_data()
  # OD = OpenData("tweet", 10000, True)
  get_labeled_data_5cate()