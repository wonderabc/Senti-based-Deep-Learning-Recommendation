# 训练情感分类模型（use keras)
# 2023.2.20 用bert4keras加载预训练模型，目前用keras-bert加载会出现一些模型权重随机（没有加载成功）的错误。目前似乎也没有解决这个问题，问题可能出在keras。。。
# ref: https://blog.csdn.net/asialee_bird/article/details/102747435
# 1007 样本maxlen调整（限制单词数maxlen=256）；选择wwm_uncased_L-24_H-1024_A-16（BERT-Large）作为base model进行训练。
import os
import tensorflow as tf
import pickle
import random
import time
from collections import defaultdict
import numpy as np
from gensim.corpora import Dictionary
from gensim.models import Word2Vec
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import KFold, train_test_split
from tensorflow.python.data import Dataset
from tensorflow.python.keras import Input, Model
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.python.keras.layers import Lambda, Dense

from sentiment_analysis.BERT_ProcessData import glue_convert_examples_to_features, glue_processors
from sentiment_analysis.generate_corpus import get_labeled_data, get_labeled_data_5cate, get_data, GetOpenData
from sentiment_analysis.train_sentiment_model_others import clean, print_test_report
from transformers import (
    BertConfig,
    BertForSequenceClassification,
    BertTokenizer,
    TFBertForSequenceClassification,
)

now_path = os.path.dirname(os.path.abspath(__file__))
root = now_path  # 当前所在目录的路径

# timeinfo = time.strftime("%Y_%m_%d_%H_%M_%S",time.localtime())  # 获取日期时间戳，方便保存模型
timeinfo = time.strftime("%Y_%m_%d",time.localtime())  # 获取日期时间戳，方便保存模型
date = "0319"
# BERT预训练模型路径
# config_path = r"D:\Workspace\workspace\BERT_solution\uncased_L-12_H-768_A-12\bert_config.json"
# checkpoint_path = r"D:\Workspace\workspace\BERT_solution\uncased_L-12_H-768_A-12\bert_model.ckpt"
# dict_path = r"D:\Workspace\workspace\BERT_solution\uncased_L-12_H-768_A-12\vocab.txt"
config_path = "F:/pretrained_model/bert-base-uncased/config.json"
dict_path = "F:/pretrained_model/bert-base-uncased/vocab.txt"
pretrained_model_path = r"F:/pretrained_model/bert-base-uncased"

# load tokenizer from pretrained model
def get_tokendict():
  dic = {}
  f = open(dict_path, encoding="utf8", mode="r")
  for line in f.readlines():
    token = line.strip()
    dic[token] = len(dic)
  return dic

tokenizer = BertTokenizer.from_pretrained(dict_path)

# LSTM 基于旧模型继续训练
have_old_model = False
oldmodel_path = "./model/0317_conti_sentiment_analysis_lstm.hdf5"

# 使用BERT-Large, Uncased (Whole Word Masking)
# config_path = r"F:\【BERT】\预训练模型\wwm_uncased_L-24_H-1024_A-16\bert_config.json"
# checkpoint_path = r"F:\【BERT】\预训练模型\wwm_uncased_L-24_H-1024_A-16\bert_model.ckpt"
# dict_path = r"F:\【BERT】\预训练模型\wwm_uncased_L-24_H-1024_A-16\vocab.txt"
bert_batchsize = 32

glovedict_path = root + "/dict/glove.6B.300d.txt"  # 用于LSTM embedding
word2vecdict_path = root + "/dict/word2vec.dict"  # word2vec embedding
maxlen = 256
lstm_dim = 300  # 用glove 300d向量 / word2vec向量化也规定为300d
lstm_batchsize = 32

# BERT 模型存储路径
BERT_model_path = root + "/bert_dump/" + timeinfo + "_sentiment_analysis_BERT_"  # 后缀+数据集名称
# LSTM 模型存储路径
LSTM_model_path = root + '/model/LSTM/' + timeinfo + '_sentiment_analysis_lstm_'  # 后缀+数据集名称

def seq_padding(X, padding=0):  # 让每条文本的长度相同，用0填充
    L = [len(x) for x in X]
    ML = max(L)
    return np.array([
        np.concatenate([x, [padding] * (ML - len(x))]) if len(x) < ML else x for x in X
    ])


class data_generator:
    def __init__(self, data, batch_size=bert_batchsize, shuffle=True):
        self.data = data
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.steps = len(self.data) // self.batch_size
        if len(self.data) % self.batch_size != 0:
            self.steps += 1

    def __len__(self):
        return self.steps

    def __iter__(self):
        while True:
            idxs = list(range(len(self.data)))
            if self.shuffle:
                np.random.shuffle(idxs)
            X1, X2, Y = [], [], []
            for i in idxs:
                d = self.data[i]
                # text = d[0][:maxlen]
                text = " ".join(d[0].split()[:maxlen])
                x1, x2 = tokenizer.encode(first=text)
                y = d[1]
                X1.append(x1)
                X2.append(x2)
                Y.append([y])
                if len(X1) == self.batch_size or i == idxs[-1]:
                    X1 = seq_padding(X1)
                    X2 = seq_padding(X2)
                    Y = seq_padding(Y)
                    # yield [X1, X2], Y[:, 0]
                    yield [X1, X2], Y[:, 0, :]
                    X1, X2, Y = [], [], []


def build_bert(nclass):  # 构建bert模型
  config = BertConfig.from_pretrained(config_path, num_labels=nclass)
  model = TFBertForSequenceClassification.from_pretrained(pretrained_model_path, config=config)  # 直接封装了全连接层+激活函数
  opt = tf.keras.optimizers.Adam(learning_rate=1e-5, epsilon=1e-8)
  loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
  metric = tf.keras.metrics.SparseCategoricalAccuracy("accuracy")
  model.compile(optimizer=opt, loss=loss, metrics=[metric])
  return model


class BERT_classifier:  # BERT分类器
  def __init__(self, OriginDataset=None, CateNum=3, dataname="amazon"):
    self.dataname = dataname
    if OriginDataset is None:  # 不传数据集的情况，默认获取Amazon数据集，TrainSentimentModels类会保证传OriginDataset
      dataset, negset, posset, neuset, maxlen = get_labeled_data()  # 获取训练集
      self.x_train, self.y_train, self.x_test, self.y_test = get_data(negset, posset, neuset, 0.9)
      # 恢复分词前句子
      self.x_train = [" ".join(x) for x in self.x_train]
      self.x_test = [" ".join(x) for x in self.x_test]
    else:
      self.dataset = OriginDataset
      self.x_train, self.y_train, self.x_test, self.y_test = GetOpenData(self.dataset, 0.9)  # 划分训练集、测试集
      # 无需对文本做分词处理
    self.CateNum = CateNum
    self.model = build_bert(self.CateNum)
    self.labellist = []
    for i in range(self.CateNum):
      self.labellist.append(i)

  def train_model(self):
    processor = glue_processors["user"]()  # 用于数据处理
    train_examples = processor.get_train_examples_by_list(self.x_train, self.y_train)
    valid_examples = processor.get_dev_examples_by_list(self.x_test, self.y_test)
    train_dataset = glue_convert_examples_to_features(train_examples, tokenizer, maxlen, "user", label_list=self.labellist)
    valid_dataset = glue_convert_examples_to_features(valid_examples, tokenizer, maxlen, "user", label_list=self.labellist)
    train_dataset = train_dataset.shuffle(len(train_examples)).batch(bert_batchsize).repeat(-1)
    valid_dataset = valid_dataset.batch(bert_batchsize)
    early_stopping = EarlyStopping(monitor='val_acc', patience=3)  # 早停法，防止过拟合
    plateau = ReduceLROnPlateau(monitor="val_acc", verbose=1, mode='max', factor=0.5,
                                patience=2)  # 当评价指标不再提升时，减少学习率
    model_path = BERT_model_path + self.dataname + "_" + str(self.CateNum) + ".hdf5"
    checkpoint = ModelCheckpoint(model_path,
                                 monitor='val_acc', verbose=2,
                                 save_best_only=True, mode='max', save_weights_only=True)  # 保存最好的模型，增加时间信息
    train_steps = len(train_examples) // bert_batchsize
    if train_steps * bert_batchsize < len(train_examples):
      train_steps += 1
    valid_steps = len(valid_examples) // bert_batchsize
    if valid_steps * bert_batchsize < len(valid_examples):
      valid_steps += 1
    self.model.fit_generator(
      train_dataset,
      steps_per_epoch=train_steps,
      epochs=5,
      validation_data=valid_dataset,
      validation_steps=valid_steps,
      callbacks=[early_stopping, plateau, checkpoint]
    )
    test_model_pred = self.model.predict_generator(valid_dataset, steps=valid_steps, verbose=1)[0]
    test_true = [x.label for x in valid_examples]
    test_pred = [np.argmax(label) for label in test_model_pred]
    # 评测模型性能
    accuracy, precision, recall, f1 = print_test_report(test_true, test_pred)

  def DoEvaluate(self, x_test, y_test, model_path):
    if x_test is None:
      x_test = self.x_test
      y_test = self.y_test
    processor = glue_processors["user"]()  # 用于数据处理
    valid_examples = processor.get_dev_examples_by_list(x_test, y_test)
    valid_dataset = glue_convert_examples_to_features(valid_examples, tokenizer, maxlen, "user", label_list=self.labellist)
    valid_dataset = valid_dataset.batch(bert_batchsize)
    valid_steps = len(valid_examples) // bert_batchsize
    if valid_steps * bert_batchsize < len(valid_examples):
      valid_steps += 1
    self.model.load_weights(model_path)
    test_model_pred = self.model.predict_generator(valid_dataset, steps=valid_steps, verbose=1)[0]
    test_true = [x.label for x in valid_examples]
    test_pred = [np.argmax(y) for y in test_model_pred]
    accuracy, precision, recall, f1 = print_test_report(test_true, test_pred)


def main():
    build_bert(3)


if __name__ == "__main__":
    main()


