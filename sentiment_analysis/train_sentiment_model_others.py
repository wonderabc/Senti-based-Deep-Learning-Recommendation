# 情感分类模型 NB / SVM
import copy
import os
import random
import time
import numpy as np
from bert_serving.client import BertClient
from gensim.models import word2vec, Word2Vec
from pandas import DataFrame
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report, f1_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
import joblib
from sklearn.svm import SVC
import multiprocessing as mp
from multiprocessing import Process
from tqdm import tqdm
from sentiment_analysis.DicBased import SentimentAnalysis
from sentiment_analysis.generate_corpus import get_labeled_data, get_data, GetOpenData
from bert_embedding import BertEmbedding  # 获取BERT的token embedding

now_path = os.path.dirname(os.path.abspath(__file__))
root = now_path  # 当前所在目录的路径
# timeinfo = time.strftime("%Y_%m_%d_%H_%M_%S",time.localtime())  # 获取日期时间戳，方便保存模型
timeinfo = time.strftime("%Y_%m_%d",time.localtime())  # 获取日期时间戳，方便保存模型
# svmmodel_path = root + "/model/" +timeinfo + "_SVMmodel.pkl"  # BERT作为文本向量化方法
# nbmodel_path = root + "/model/" + timeinfo + "_NBmodel.pkl"
# svmmodel_w2v_path = root + "/model/" + timeinfo + "_SVM_w2v_model.pkl"

# 增加数据集名称作为模型存储的后缀
svmmodel_path = root + "/model/" +timeinfo + "_SVMmodel_"  # BERT作为文本向量化方法
nbmodel_path = root + "/model/" + timeinfo + "_NBmodel_"
svmmodel_w2v_path = root + "/model/" + timeinfo + "_SVM_w2v_model_"
dic_path = root + "/data/SentiWordNet.txt"  # 情感词典的路径
DicThreshold = 1e-7  # 判断是否为中性的阈值
stopwordpath = root + "/stopwords.txt"

def start_bert_service():  # start bert encoding
    # 加载英文预训练模型
    cmd = "D: && cd D:\\Workspace\\workspace\\BERT_solution " \
          "&& bert-serving-start -model_dir uncased_L-12_H-768_A-12"  # 可以指定端口，有时会出现端口被占用
    status = os.system(cmd)  # 0 success / 1 fail
    # q.put(status)
    if status != 0:
        print("Error when starting bert encoding service!")


def get_stopwords():
    with open(stopwordpath, "r", encoding="utf8") as f:
        stop_word = f.read()
    stop_word_list = stop_word.strip().split("\n")
    custom_stopword = [word for word in stop_word_list if len(word) > 0]
    return custom_stopword


def build_sentence_vector(content, size, model):
  vec = np.zeros(size).reshape((1, size))
  cnt = 0
  for word in content:
    try:
      vec += model.wv[word].reshape((1, size))
      cnt += 1
    except KeyError:
      continue
  if cnt != 0:
    vec /= cnt
  return vec

def clean(content):  # 清洗待分类文本
  return content.replace(",", " ").replace(".", " ").replace(";", " ").split(" ")

def print_test_report(y_test, y_pred):  # 打印测试集性能
  print("测试集性能如下：")
  accuracy = accuracy_score(y_test, y_pred)
  precision = precision_score(y_test, y_pred, average="macro")
  recall = recall_score(y_test, y_pred, average="macro")
  f1 = f1_score(y_test, y_pred, average="macro")
  # print(classification_report(y_test, y_pred))
  print("测试集的准确率是{}，精确率是{}，召回率是{}，f1_score是{}。".format(accuracy, precision, recall, f1))
  return accuracy, precision, recall, f1


class NB_classify:
    def __init__(self, OriginDataset=None, dataname="amazon", CateNum=3):
      self.CateNum = CateNum
      self.dataname = dataname
      if OriginDataset is None:  # 不传数据集的情况，默认获取Amazon数据集，TrainSentimentModels类会保证传OriginDataset
        dataset, negset, posset, neuset, maxlen = get_labeled_data()  # 获取训练集
        self.x_train, self.y_train, self.x_test, self.y_test = get_data(negset, posset, neuset, 0.9)
        self.x_train = [" ".join(x) for x in self.x_train]  # 不需要分词
        self.x_test = [" ".join(x) for x in self.x_test]
        self.stopwords = get_stopwords()
      else:
        self.dataset = OriginDataset
        self.x_train, self.y_train, self.x_test, self.y_test = GetOpenData(self.dataset, 0.9)  # 划分训练集、测试集
        # self.x_train = [clean(x) for x in self.x_train]
        # self.x_test = [clean(x) for x in self.x_test]
      self.y_train = [np.argmax(y) for y in self.y_train]
      self.y_test = [np.argmax(y) for y in self.y_test]
      self.stopwords = get_stopwords()
      self.model = None  # 初始化时不加载模型

    def train_model(self):
        max_df = 0.5
        min_df = 3
        vect = CountVectorizer(max_df=max_df, min_df=min_df, token_pattern=u'(?u)\\b[^\\d\\W]\\w+\\b',
                               stop_words=frozenset(self.stopwords))
        # term_matrix = DataFrame(vect.fit_transform(self.x_train).toarray(), columns=vect.get_feature_names())
        # print(term_matrix)
        nb = MultinomialNB()
        # nb.fit(term_matrix, y_train)
        pipe = make_pipeline(vect, nb)
        # print(self.x_train)
        pipe.fit(self.x_train, self.y_train)
        # print(pipe)
        # print("交叉测试准确率：")
        # print(cross_val_score(pipe, self.x_train, self.y_train, cv=5, scoring='accuracy').mean())
        model_path = nbmodel_path + self.dataname + "_" + str(self.CateNum) + ".pkl"
        joblib.dump(pipe, model_path)
        y_pred = list(pipe.predict(self.x_test))
        accuracy, precision, recall, f1 = print_test_report(self.y_test, y_pred)

    def DoEvaluate(self, x_test, y_test, model_path):
      """
      :param x_test: 文本，无需向量化
      :param y_test: label 非onehot形式
      :param model_path: 模型路径，在外部给定
      :return:
      """
      if x_test is None:
        x_test = self.x_test
        y_test = self.y_test
      self.model = joblib.load(model_path)  # 加载模型
      y_pred = list(self.model.predict(x_test))
      accuracy, precision, recall, f1 = print_test_report(y_test, y_pred)

class SVM_classify:  # 用BERT得到词向量的SVM
    def __init__(self, OriginDataset=None, dataname="amazon", CateNum=3):
        self.CateNum = CateNum
        self.dataname = dataname
        # q = mp.Queue()  # 进程结果池
        # 跑benchmark时，人工在外部启动
        # self.bert_service = Process(target=start_bert_service, args=())
        # self.bert_service.start()
        # time.sleep(90)  # bert-service 90s的启动时间
        if OriginDataset is None:  # 不传数据集的情况，默认获取Amazon数据集，TrainSentimentModels类会保证传OriginDataset
          dataset, negset, posset, neuset, maxlen = get_labeled_data()  # 获取训练集
          self.x_train, self.y_train, self.x_test, self.y_test = get_data(negset, posset, neuset, 0.9)
          self.x_train = [" ".join(x) for x in self.x_train]  # 不需要分词
          self.x_test = [" ".join(x) for x in self.x_test]
        else:
          self.dataset = OriginDataset
          self.x_train, self.y_train, self.x_test, self.y_test = GetOpenData(self.dataset, 0.9)  # 划分训练集、测试集
          # self.x_train = [clean(x) for x in self.x_train]
          # self.x_test = [clean(x) for x in self.x_test]
        self.y_train = [np.argmax(y) for y in self.y_train]
        self.y_test = [np.argmax(y) for y in self.y_test]
        self.x_train, self.x_test = self.load_file_and_preprocessing()  # BERT encode后的训练集、测试集
        self.stopwords = get_stopwords()  # 获取停用词
        self.model = None  # 初始化时不加载模型

    def load_file_and_preprocessing(self):
        # 将样本的文本内容转化为向量
        # bc = BertClient(check_length=False)
        # x_train = bc.encode(self.x_train)  # 转换成向量
        # x_test = bc.encode(self.x_test)

        # tf2 无法使用bert-serving-server/client包
        bc = BertEmbedding()
        x_train, x_test = [], []
        EmbeddingResults = bc(self.x_train)
        for i in tqdm(range(len(EmbeddingResults))):
          EmbeddingResult = EmbeddingResults[i]
          x_train.append(np.mean(EmbeddingResult[1], axis=0))
        EmbeddingResults = bc(self.x_test)
        for i in tqdm(range(len(EmbeddingResults))):
          EmbeddingResult = EmbeddingResults[i]
          x_test.append(np.mean(EmbeddingResult[1], axis=0))
        return x_train, x_test

    def train_model(self):  # 训练svm模型
        print("开始训练！")
        # print(self.x_train)
        clf = SVC(kernel="rbf", verbose=True)
        clf.fit(self.x_train, self.y_train)
        print("训练完成。\n测试集表现如下：")
        # print(clf.score(self.x_test, self.y_test))  # 测试集表现
        model_path = svmmodel_path + self.dataname + "_" + str(self.CateNum) + ".pkl"
        joblib.dump(clf, model_path)
        # self.bert_service.terminate()
        y_pred = clf.predict(self.x_test)
        accuracy, precision, recall, f1 = print_test_report(self.y_test, y_pred)

    def DoEvaluate(self, x_test, y_test, model_path):
      """
      :param x_test: 需要向量化后才能做预测
      :param y_test: label 非onehot形式
      :param model_path: 模型路径，在外部给定
      :return:
      """
      if x_test is None:
        x_test = self.x_test
        y_test = self.y_test
      self.model = joblib.load(model_path)  # 加载模型
      y_pred = list(self.model.predict(x_test))
      accuracy, precision, recall, f1 = print_test_report(y_test, y_pred)

class SVM_classify_usew2v:
  def __init__(self, OriginDataset=None, dataname="amazon", CateNum=3):
    self.dataname = dataname
    self.CateNum = CateNum
    if OriginDataset is None:  # 不传数据集的情况，默认获取Amazon数据集，TrainSentimentModels类会保证传OriginDataset
      dataset, negset, posset, neuset, maxlen = get_labeled_data()  # 获取训练集
      self.x_train, self.y_train, self.x_test, self.y_test = get_data(negset, posset, neuset, 0.9)
    else:
      self.dataset = OriginDataset
      self.x_train, self.y_train, self.x_test, self.y_test = GetOpenData(self.dataset, 0.9)  # 划分训练集、测试集
      self.x_train = [clean(x) for x in self.x_train]
      self.x_test = [clean(x) for x in self.x_test]
    self.y_train = [np.argmax(y) for y in self.y_train]
    self.y_test = [np.argmax(y) for y in self.y_test]
    self.stopwords = get_stopwords()
    self.model = None  # 初始化时不加载模型
    self.train_vecs_path = root + "/svm_data/train_vecs.npy"
    self.test_vecs_path = root + "/svm_data/test_vecs.npy"
    self.y_train_path = root + "/svm_data/y_train.npy"
    self.y_test_path = root + "/svm_data/y_test.npy"
    np.save(root + "/svm_data/y_train.npy", self.y_train)
    np.save(root + "/svm_data/y_test.npy", self.y_test)
    self.get_train_vecs()

  def get_train_vecs(self):  # 计算词向量并保存为train_vecs.npy,test_vecs.npy
    n_dim = 100
    w2vmodel = Word2Vec.load(root + '/svm_data/w2v_100d_model.pkl')  # 直接加载
    # 加载后增量训练
    # w2vmodel.build_vocab(self.x_train)
    # w2vmodel.train(self.x_train, total_examples=w2vmodel.corpus_count, epochs=2)
    # print(w2vmodel.wv["love"])
    train_vecs = np.concatenate([build_sentence_vector(z, n_dim, w2vmodel) for z in self.x_train])
    np.save(root + "/svm_data/train_vecs.npy", train_vecs)
    # w2vmodel.train(self.x_test, total_examples=w2vmodel.corpus_count, epochs=2)
    # w2vmodel.save("./svm_data/w2v_model.pkl")
    test_vecs = np.concatenate([build_sentence_vector(z, n_dim, w2vmodel) for z in self.x_test])
    np.save(root + "/svm_data/test_vecs.npy", test_vecs)

  def train_model(self):
    train_vecs = np.load(self.train_vecs_path)
    y_train = np.load(self.y_train_path)
    test_vecs = np.load(self.test_vecs_path)
    y_test = np.load(self.y_test_path)
    print("开始训练SVM分类模型！")
    clf = SVC(kernel="rbf", verbose=False)
    clf.fit(train_vecs, y_train)
    # print(clf.score(test_vecs, y_test))
    print("存储训练好的SVM分类模型……")
    model_w2v_path = svmmodel_w2v_path + self.dataname + "_" + str(self.CateNum) + ".pkl"
    joblib.dump(clf, model_w2v_path)
    y_pred = clf.predict(test_vecs)
    accuracy, precision, recall, f1 = print_test_report(y_test, y_pred)

  def DoEvaluate(self, x_test, y_test, model_path):
    """

    :param x_test: 需要向量化才能做预测
    :param y_test: label 非onehot形式
    :param model_path: 模型路径，在外部给定
    :return:
    """
    if x_test is None:
      x_test = self.x_test
      y_test = self.y_test
    self.model = joblib.load(model_path)  # 加载模型
    n_dim = 100
    w2vmodel = Word2Vec.load(root + '/svm_data/w2v_100d_model.pkl')  # 直接加载
    test_vecs = np.concatenate([build_sentence_vector(z, n_dim, w2vmodel) for z in x_test])
    y_pred = list(self.model.predict(test_vecs))
    accuracy, precision, recall, f1 = print_test_report(y_test, y_pred)

class DicBased_Classify:  # 词典法分类器
  def __init__(self, OriginDataset=None, CateNum=3):  # 词典法需额外指定类别数，仅能handle 二分类和三分类问题
    self.threshold = DicThreshold
    self.CateNum = CateNum
    if OriginDataset is None:  # 不传数据集的情况，默认获取Amazon数据集，TrainSentimentModels类会保证传OriginDataset
      dataset, negset, posset, neuset, maxlen = get_labeled_data()  # 获取训练集
      self.x_train, self.y_train, self.x_test, self.y_test = get_data(negset, posset, neuset, 0.9)
      self.x_train = [" ".join(x) for x in self.x_train]  # 不需要分词
      self.x_test = [" ".join(x) for x in self.x_test]
    else:
      self.dataset = OriginDataset
      self.x_train, self.y_train, self.x_test, self.y_test = GetOpenData(self.dataset, 0.9)  # 划分训练集、测试集

  def evaluate_model(self):  # 词典法无需训练，直接评测测试集的表现
    s = SentimentAnalysis(dic_path, weighting='geometric')
    y_test = [np.argmax(y) for y in self.y_test]
    y_pred = []
    for x in self.x_test:
      score = s.score(x)
      if self.CateNum == 3:
        if abs(score) <= self.threshold:
          y_pred.append(2)
        elif score > 0:
          y_pred.append(1)
        else:
          y_pred.append(0)
      elif self.CateNum == 2:
        if score > 0:
          y_pred.append(1)
        else:
          y_pred.append(0)
    accuracy, precision, recall, f1 = print_test_report(y_test, y_pred)

  def DoEvaluate(self, x_test, y_test):
    if x_test is None:
      x_test = self.x_test
      y_test = self.y_test
    s = SentimentAnalysis(dic_path, weighting='geometric')
    y_test = [np.argmax(y) for y in y_test]
    y_pred = []
    for x in x_test:
      score = s.score(x)
      if self.CateNum == 3:
        if abs(score) <= self.threshold:
          y_pred.append(2)
        elif score > 0:
          y_pred.append(1)
        else:
          y_pred.append(0)
      elif self.CateNum == 2:
        if score > 0:
          y_pred.append(1)
        else:
          y_pred.append(0)
    accuracy, precision, recall, f1 = print_test_report(y_test, y_pred)

def evaluate_models():  # 评测模型
    print("NBmodel:")
    nb = NB_classify()
    nbmodel = joblib.load(nbmodel_path)
    y_pred = list(nbmodel.predict(nb.x_test))
    accuracy = accuracy_score(nb.y_test, y_pred)
    precision = precision_score(nb.y_test, y_pred, average="macro")
    recall = recall_score(nb.y_test, y_pred, average="macro")
    f1 = f1_score(nb.y_test, y_pred, average="macro")
    print("测试集的准确率是{}，精确率是{}，召回率是{}，f1_score是{}。".format(accuracy, precision, recall, f1))
    # print(classification_report(nb.y_test, y_pred))

    print("SVMmodel:")
    svm = SVM_classify()
    svmmodel = joblib.load(svmmodel_path)
    # print(svmmodel.score(svm.x_test, svm.y_test))
    y_pred = list(svmmodel.predict(svm.x_test))
    accuracy = accuracy_score(svm.y_test, y_pred)
    precision = precision_score(svm.y_test, y_pred, average="macro")
    recall = recall_score(svm.y_test, y_pred, average="macro")
    f1 = f1_score(svm.y_test, y_pred, average="macro")
    print("测试集的准确率是{}，精确率是{}，召回率是{}，f1_score是{}。".format(accuracy, precision, recall, f1))


if __name__ == "__main__":
  # 使用外部语料库预训练word2vec模型
  # sentences = word2vec.Text8Corpus('./data/text8')  # 使用外部语料库
  # w2vmodel = Word2Vec(sentences, min_count=3, vector_size=300)  # 初始化模型
  # w2v_100d_path = root + "/svm_data/w2v_100d_model.pkl"
  # w2v_300d_path = root + "/svm_data/w2v_300d_model.pkl"
  # w2vmodel.save(w2v_300d_path)

  # w2vmodel2 = Word2Vec(sentences, min_count=3)  # 初始化模型
  # w2vmodel2.save(w2v_100d_path)

  # model_NB = NB_classify()
  # model_NB.train_model()
  # model_SVM = SVM_classify()
  # model_SVM.train_model()
  # model_SVM = SVM_classify_usew2v()
  # model_SVM.train_model()
  # evaluate_models()  # 评测训练好的模型

  # bert encode 验证
  # bert_service = Process(target=start_bert_service, args=())
  # bert_service.start()
  # time.sleep(90)  # bert-service 90s的启动时间
  # bc = BertClient(check_length=False)
  # print(bc.encode(["I love you."]))
  print(1)




