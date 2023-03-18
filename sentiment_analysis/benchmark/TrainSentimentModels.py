# 在不同的数据集，用不同的方法训练情感分类模型
# 数据规模：二分类 2300 * 2；三分类 2200 * 3；五分类 1500 * 5（不适用于自己清洗的数据集）
import random
import time
import numpy as np
from collections import defaultdict

from keras.backend import clear_session
from keras.utils import to_categorical
from sentiment_analysis.generate_corpus import OpenData, get_labeled_data, get_labeled_data_5cate, Transform3to2, \
  Transform5to2, Transform5to3
from sentiment_analysis.train_sentiment_model import BiLSTM_classifier
from sentiment_analysis.train_sentiment_model_tfversion import BERT_classifier
from sentiment_analysis.train_sentiment_model_others import SVM_classify_usew2v, NB_classify, SVM_classify, \
  DicBased_Classify, start_bert_service
from multiprocessing import Process

NoScaleLimit = False # 数据集规模需指定
datasources = ["amazon", "sst5", "imdb", "yelp", "tweet"]  # 数据集
# datasources = ["yelp", "tweet"]
catedict = {"amazon": [2, 3], "sst5": [2, 3, 5], "imdb": [2], "yelp": [2, 3, 5], "tweet": [2, 3]}  # 各数据集对应的类别数
# catedict = {"amazon": [3], "sst5": [2, 3, 5], "imdb": [2], "yelp": [2], "tweet": [2, 3]}  # 各数据集对应的类别数
scaledict = {2: 4600, 3: 6600, 5: 7500}  # x分类任务对应的样本规模
modelpath = {"svm": "D:/Workspace/workspace/Steam_Recommend/sentiment_analysis/model/2023_02_23_SVM_w2v_model_",
             "svm-BERT": "D:/Workspace/workspace/Steam_Recommend/sentiment_analysis/model/2023_02_23_SVMmodel_",
             "nb": "D:/Workspace/workspace/Steam_Recommend/sentiment_analysis/model/2023_02_23_NBmodel_",
             "LSTM": "E:/PaperModels/LSTM/2023_02_19_sentiment_analysis_lstm_",
             "LSTM-glove": "E:/PaperModels/LSTM_glove/2023_02_18_sentiment_analysis_lstm_",
             "LSTM-novec": "E:/PaperModels/LSTM_novec/2023_02_24_sentiment_analysis_lstm_",
             "BERT": "E:/PaperModels/bert_dump/2023_02_22_sentiment_analysis_BERT_"}
modeltypes = ["svm", "nb", "dic", "LSTM", "BERT"]  # benchmark模型
# modeltypes = ["svm"]  # benchmark模型
# modeltypes = ["LSTM"]  # benchmark模型

def GetPolarity(label, num):  # 依据array形式的y获得类别标签（0，1，2）
  for i in range(num):
    if label[i] == 1:
      return i

class TrainSentimentModels:
  def __init__(self, dataname="imdb", scale=6600, CateNum=3):
    # dataname指定数据集，scale和CateNum共同指定样本规模
    self.CateNum = CateNum
    self.dataname = dataname
    if dataname == "amazon" or dataname == "sst5":
      # 此时需要根据样本规模要求重新抽样
      if dataname == "amazon":
        self.dataset, negset, posset, neuset, maxlen = get_labeled_data()
      elif dataname == "sst5":
        Trainset, Devset, Testset = get_labeled_data_5cate()
        Trainset, Devset, Testset = list(Trainset), list(Devset), list(Testset)
        self.dataset = Trainset + Devset + Testset
      self.DataSet = self.TransformSomeData(dataname, scale, CateNum)
    else:
      # 获取公开数据集
      GetData = OpenData(dataname, scale, NoScaleLimit, CateNum)  # 已包含极性转换过程
      self.DataSet = GetData.dataset  # 获得数据集
    # self.DataSet的格式保证是[(content, onehot类别向量)]

  def TransformSomeData(self, dataname, scale, CateNum):  # amazon及sst5数据集的抽样及极性转换过程，原先数据集没有进行抽样过程；OpenData类已经预先实现了抽样+极性转换
    DataSet = defaultdict(list)
    CateSize = scale // CateNum  # 每一类的规模

    for x, y in self.dataset:
      polarity = int(np.argmax(y))
      if not isinstance(polarity, int):  # 判断是否已转换为int的类别
        continue
      if dataname == "amazon":
        if CateNum == 2:
          polarity = Transform3to2(polarity)
      elif dataname == "sst5":
        if CateNum == 2:
          polarity = Transform5to2(polarity)
        elif CateNum == 3:
          polarity = Transform5to3(polarity)
      if polarity == -1:
        continue
      DataSet[polarity].append((x, to_categorical(polarity, CateNum)))
    for key in DataSet:
      DataSet[key] = random.sample(DataSet[key], CateSize)
    return DataSet

  def TrainModels(self, modelname="svm"):
    if modelname == "svm":
      # model = SVM_classify_usew2v(self.DataSet, self.dataname, self.CateNum)
      # model.train_model()
      print("使用BERT 向量化：")
      model = SVM_classify(self.DataSet, self.dataname, self.CateNum)  # 用BERT作为文本向量化工具
      model.train_model()
    elif modelname == "nb":
      model = NB_classify(self.DataSet, self.dataname, self.CateNum)
      model.train_model()
    elif modelname == "dic":
      if self.CateNum > 3:  # 不能处理五分类问题
        print("词典法无法处理{}分类问题，请重新确认参数！".format(self.CateNum))
        return
      model = DicBased_Classify(self.DataSet, self.CateNum)
      model.evaluate_model()
    elif modelname == "LSTM":
      model = BiLSTM_classifier(self.DataSet, self.CateNum, self.dataname)
      model.train_model()
    elif modelname == "BERT":
      model = BERT_classifier(self.DataSet, self.CateNum, self.dataname)
      model.train_model()

  def Evaluate(self, x=None, y=None, modelname="svm"):  # 基于给定的数据集评测模型的性能
    if modelname == "svm":
      model = SVM_classify_usew2v(self.DataSet, self.dataname, self.CateNum)
      path = modelpath[modelname] + self.dataname + "_" + str(self.CateNum) + ".pkl"
      print("Evaluating model: {}......".format(path))
      model.DoEvaluate(x, y, path)
      # print("使用BERT 向量化：")
      # path = modelpath[modelname + "-BERT"] + self.dataname + "_" + str(self.CateNum) + ".pkl"
      # print("Evaluating model: {}......".format(path))
      # model = SVM_classify(self.DataSet, self.dataname)  # 用BERT作为文本向量化工具
      # model.DoEvaluate(x, y, path)
    elif modelname == "nb":
      model = NB_classify(self.DataSet, self.dataname, self.CateNum)
      path = modelpath[modelname] + self.dataname + "_" + str(self.CateNum) + ".pkl"
      print("Evaluating model: {}......".format(path))
      model.DoEvaluate(x, y, path)
    elif modelname == "dic":
      if self.CateNum > 3:  # 不能处理五分类问题
        print("词典法无法处理{}分类问题，请重新确认参数！".format(self.CateNum))
        return
      print("Evaluating Dic-based method......")
      model = DicBased_Classify(self.DataSet, self.CateNum)
      model.DoEvaluate(x, y)
    elif modelname == "LSTM":
      model = BiLSTM_classifier(self.DataSet, self.CateNum, self.dataname, "novec")
      path = modelpath[modelname + "-novec"] + self.dataname + "_" + str(self.CateNum) + ".hdf5"
      print("Evaluating model: {}......".format(path))
      model.DoEvaluate(x, y, path)

      model = BiLSTM_classifier(self.DataSet, self.CateNum, self.dataname, "glove")
      path = modelpath[modelname + "-glove"] + self.dataname + "_" + str(self.CateNum) + ".hdf5"
      print("Evaluating model: {}......".format(path))
      model.DoEvaluate(x, y, path)

      model = BiLSTM_classifier(self.DataSet, self.CateNum, self.dataname, "word2vec")
      path = modelpath[modelname] + self.dataname + "_" + str(self.CateNum) + ".hdf5"
      print("Evaluating model: {}......".format(path))
      model.DoEvaluate(x, y, path)
    elif modelname == "BERT":
      model = BERT_classifier(self.DataSet, self.CateNum, self.dataname)
      path = modelpath[modelname] + self.dataname + "_" + str(self.CateNum) + ".hdf5"
      print("Evaluating model: {}......".format(path))
      model.DoEvaluate(x, y, path)


def EvaluateModels():  # 多数据集上各个模型的benchmark实验
  # 跑SVM-BERT的benchmark，人工启动BERT embedding服务
  # tf2无法启动该服务
  # bert_service = Process(target=start_bert_service, args=())
  # bert_service.start()
  # time.sleep(90)  # bert-service 90s的启动时间

  # train = TrainSentimentModels(dataname="amazon", scale=4600, CateNum=2)
  # train = TrainSentimentModels(dataname="amazon", scale=6600, CateNum=3)
  # model = BERT_classifier(train.DataSet, train.CateNum, train.dataname)
  # model.DoEvaluate(None, None, "E:/PaperModels/bert_dump/2023_02_22_sentiment_analysis_BERT_amazon_2.hdf5")
  defaultdata = True
  train = TrainSentimentModels(dataname="amazon", scale=4600, CateNum=2)
  dataset_cate2 = train.DataSet
  train = TrainSentimentModels(dataname="amazon", scale=6600, CateNum=3)
  dataset_cate3 = train.DataSet
  for datasource in datasources:
    catelist = catedict[datasource]
    for cate in catelist:
      scale = scaledict[cate]
      train = TrainSentimentModels(dataname=datasource, scale=scale, CateNum=cate)
      if defaultdata:  # 统一数据集
        if cate == 2:
          train.DataSet = dataset_cate2
        elif cate == 3:
          train.DataSet = dataset_cate3
        else:
          print("类别数为5，无法用AMAZON数据集评估！")
          continue
      for model in modeltypes:
        print("数据集：{}，类别总数：{}，模型类别：{}。".format(datasource, cate, model))
        # train.TrainModels(modelname=model)
        train.Evaluate(x=None, y=None, modelname=model)

  # 评估特定推荐领域数据集的模型表现

if __name__ == "__main__":
  EvaluateModels()
