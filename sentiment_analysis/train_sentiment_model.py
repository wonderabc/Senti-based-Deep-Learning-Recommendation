# 训练情感分类模型（use keras)
# tf 2.2.0 + keras 2.3.1 + keras-bert 0.89.0
# ref: https://blog.csdn.net/asialee_bird/article/details/102747435
# 1007 样本maxlen调整（限制单词数maxlen=256）；选择wwm_uncased_L-24_H-1024_A-16（BERT-Large）作为base model进行训练。
import os
import pickle
import random
import time
from collections import defaultdict
import numpy as np
from gensim.corpora import Dictionary
from gensim.models import Word2Vec
from keras import Input, Model, Sequential
from keras.backend import clear_session
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.engine.saving import model_from_json
from keras.layers import Lambda, Dense, Embedding, LSTM, Dropout, Activation, Bidirectional
from keras.metrics import sparse_categorical_accuracy, categorical_accuracy
from keras.optimizers import Adam
from keras.preprocessing import sequence
from keras_bert import load_trained_model_from_checkpoint, Tokenizer, get_custom_objects
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import KFold, train_test_split
from sentiment_analysis.generate_corpus import get_labeled_data, get_labeled_data_5cate, get_data, GetOpenData
from sentiment_analysis.train_sentiment_model_others import clean, print_test_report

now_path = os.path.dirname(os.path.abspath(__file__))
root = now_path  # 当前所在目录的路径

# timeinfo = time.strftime("%Y_%m_%d_%H_%M_%S",time.localtime())  # 获取日期时间戳，方便保存模型
timeinfo = time.strftime("%Y_%m_%d",time.localtime())  # 获取日期时间戳，方便保存模型
date = "0319"
# BERT预训练模型路径
config_path = r"D:\Workspace\workspace\BERT_solution\uncased_L-12_H-768_A-12\bert_config.json"
checkpoint_path = r"D:\Workspace\workspace\BERT_solution\uncased_L-12_H-768_A-12\bert_model.ckpt"
dict_path = r"D:\Workspace\workspace\BERT_solution\uncased_L-12_H-768_A-12\vocab.txt"

# LSTM 基于旧模型继续训练
have_old_model = False
oldmodel_path = "./model/0317_conti_sentiment_analysis_lstm.hdf5"

# 使用BERT-Large, Uncased (Whole Word Masking)
# config_path = r"F:\【BERT】\预训练模型\wwm_uncased_L-24_H-1024_A-16\bert_config.json"
# checkpoint_path = r"F:\【BERT】\预训练模型\wwm_uncased_L-24_H-1024_A-16\bert_model.ckpt"
# dict_path = r"F:\【BERT】\预训练模型\wwm_uncased_L-24_H-1024_A-16\vocab.txt"
bert_batchsize = 4

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
                    yield [X1, X2], Y[:, 0]
                    X1, X2, Y = [], [], []


def get_tokendict():
    dic = {}
    f = open(dict_path, encoding="utf8", mode="r")
    for line in f.readlines():
        token = line.strip()
        dic[token] = len(dic)
    return dic


token_dict = get_tokendict()
tokenizer = Tokenizer(token_dict)


def build_bert(nclass):  # 构建bert模型
    bert_model = load_trained_model_from_checkpoint(config_path, checkpoint_path, seq_len=None)  # 加载预训练模型
    for layer in bert_model.layers:
        layer.trainable = True

    x1_in = Input(shape=(None, ))
    x2_in = Input(shape=(None, ))

    x = bert_model([x1_in, x2_in])
    x = Lambda(lambda x: x[:, 0])(x)  # 取出[CLS]对应的向量用来做分类
    p = Dense(nclass, activation="softmax")(x)
    model = Model([x1_in, x2_in], p)
    model.compile(loss='categorical_crossentropy',
                       optimizer=Adam(1e-5),  # 初始学习率 1e-5
                       metrics=['acc', categorical_accuracy])
    # print(model.summary())
    return model


def build_bert_add_BiLSTM(nclass):  # 构建BERT，增加RNN层
    bert_model = load_trained_model_from_checkpoint(config_path, checkpoint_path, seq_len=None)  # 加载预训练模型
    for layer in bert_model.layers:
        layer.trainable = True

    x1_in = Input(shape=(None, ))
    x2_in = Input(shape=(None, ))

    x = bert_model([x1_in, x2_in])
    # x = Lambda(lambda x: x[:, 0])(x)  # 取出[CLS]对应的向量用来做分类
    BiLSTM_output = Bidirectional(LSTM(output_dim=256, return_sequences=False))(x)
    p = Dense(nclass, activation="softmax")(BiLSTM_output)
    model = Model([x1_in, x2_in], p)
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(5e-6),  # 初始学习率 1e-5
                  metrics=['acc', categorical_accuracy])
    print(model.summary())
    return model


def get_train_test_set(neg, pos, neu, train_rate=0.9):  # 划分训练集和测试集（9:1）
    random.seed(127)
    # train_rate = 0.9
    traindata, testdata = [], []
    traindata.extend(random.sample(neg, int(len(neg) * train_rate)))
    testdata.extend([(x, y) for x, y in neg if (x, y) not in traindata])
    traindata.extend(random.sample(pos, int(len(pos) * train_rate)))
    testdata.extend([(x, y) for x, y in pos if (x, y) not in traindata])
    traindata.extend(random.sample(neu, int(len(neu) * train_rate)))
    testdata.extend([(x, y) for x, y in neu if (x, y) not in traindata])
    # traindata = traindata[:200]  # debug用，减小训练集规模
    traindata = np.array(traindata)
    testdata = np.array(testdata)
    return traindata, testdata


def run_cv(nfold, traindata, testdata):  # 交叉验证
    kf = KFold(n_splits=nfold, shuffle=True, random_state=127).split(traindata)
    test_true = [np.argmax(y) for x, y in testdata]

    train_model_pred = np.zeros((len(traindata), 3))
    test_model_pred = np.zeros((len(testdata), 3))

    for i, (train_fold, test_fold) in enumerate(kf):
        X_train, X_valid = traindata[train_fold], traindata[test_fold]
        model = build_bert(3)
        # model = build_bert_add_BiLSTM(3)
        early_stopping = EarlyStopping(monitor='val_acc', patience=3)  # 早停法，防止过拟合
        plateau = ReduceLROnPlateau(monitor="val_acc", verbose=1, mode='max', factor=0.5,
                                    patience=2)  # 当评价指标不再提升时，减少学习率
        checkpoint = ModelCheckpoint('./bert_dump/' + 'sentiment_analysis_BERT_test' + str(i) + '.hdf5', monitor='val_acc', verbose=2,
                                     save_best_only=True, mode='max', save_weights_only=True)  # 保存最好的模型，增加时间信息
        train = data_generator(X_train, shuffle=True)
        valid = data_generator(X_valid, shuffle=True)
        test = data_generator(testdata, shuffle=False)
        # model.load_weights("./bert_dump/sentiment_analysis_1005_v3_0.hdf5")  # 加载训练好的模型参数
        model.fit_generator(
            train.__iter__(),
            steps_per_epoch=len(train),
            epochs=5,
            validation_data=valid.__iter__(),
            validation_steps=len(valid),
            callbacks=[early_stopping, plateau, checkpoint]
        )
        train_model_pred[test_fold, :] = model.predict_generator(valid.__iter__(), steps=len(valid), verbose=1)
        test_model_pred += model.predict_generator(test.__iter__(), steps=len(test), verbose=1)
        test_pred = [np.argmax(x) for x in test_model_pred]

        print('./bert_dump/' + 'sentiment_analysis_1007_uselarge_v1_' + str(i) + '.hdf5')
        print(accuracy_score(test_true, test_pred))
        # del model
        # gc.collect()  # 清理内存
        clear_session()
        # break
    return train_model_pred, test_model_pred


def get_glove_vector(path):
    f = open(path, "r", encoding="utf8")
    lines = f.readlines()
    word2idx = defaultdict(int)
    idx2vec = np.zeros((len(lines)+1, lstm_dim))
    idx = 0
    for line in lines:
        idx += 1
        items = line.strip().split()
        word = items[0].strip()
        vec = np.array([float(num) for i, num in enumerate(items) if i > 0])
        word2idx[word] = idx
        idx2vec[idx] = vec
    return word2idx, idx2vec

def Incremental_train_word2vec(sentences, n_dim):  # 增量训练Word2vec
  model = Word2Vec.load(root + '/svm_data/w2v_300d_model.pkl')  # 直接加载
  # model.build_vocab(sentences)
  # model.train(sentences, total_examples=model.corpus_count, epochs=2)  # 增量训练会导致参数不适配新的训练集
  return model

def get_word2vec_vector(n_dim, model):  # word2vec向量化
  word_vectors = model.wv
  vector_size = word_vectors.vector_size
  word2idx = defaultdict(int)
  word_size = len(word_vectors.key_to_index)
  idx2vec = np.zeros((word_size + 1, n_dim))
  idx = 0
  for key in word_vectors.key_to_index:
    idx += 1
    word2idx[key] = idx
    idx2vec[idx] = word_vectors[key].reshape((1, n_dim))
    # print(key, idx2vec[idx])
  return word2idx, idx2vec

def text2idx(dic, data):
    new_sentences = []
    for sentence in data:
        new_sen = []
        for w in sentence:
            new_sen.append(dic[w])
        new_sentences.append(new_sen)
    return np.array(new_sentences)


class LSTMdata:  # 训练LSTM需要的数据格式
    def __init__(self, data):
        self.data = data
        self.sentences = []  # 最终的句子列表，以单词列表的形式存储
        self.labels = []

    def get_sentences(self):
        sentencelist = [sentence.strip() for sentence, label in self.data]
        labellist = [label for sentence, label in self.data]
        idx = 0
        for s in sentencelist:
            wordlist = [w for w in s.split() if len(w) > 0]  # 单词列表
            if len(wordlist) == 0:
                idx += 1
                continue
            self.sentences.append(wordlist)
            self.labels.append(labellist[idx])
            idx += 1

class BiLSTM_classifier:  # 双向LSTM分类器
  def __init__(self, OriginDataset=None, CateNum=3, dataname="amazon", EmbeddingMode="word2vec"):
    """
    :param OriginDataset: 传入的数据集
    :param CateNum: 类别数
    :param dataname: 数据集名称
    :param EmbeddingMode: embedding模式 word2vec / glove / novec (直接训练Embedding)
    """
    self.dataname = dataname
    self.EmbeddingMode = EmbeddingMode
    if OriginDataset is None:  # 不传数据集的情况，默认获取Amazon数据集，TrainSentimentModels类会保证传OriginDataset
      dataset, negset, posset, neuset, maxlen = get_labeled_data()  # 获取训练集
      self.x_train, self.y_train, self.x_test, self.y_test = get_data(negset, posset, neuset, 0.9)
    else:
      self.dataset = OriginDataset
      self.x_train, self.y_train, self.x_test, self.y_test = GetOpenData(self.dataset, 0.9)  # 划分训练集、测试集
      # 需要分词
      self.x_train = [x.split() for x in self.x_train]
      self.x_test = [x.split() for x in self.x_test]
    self.CateNum = CateNum
    # self.word2idx, self.idx2vec = get_glove_vector(glovedict_path)  # GLOVE词典
    diclen = 0
    if EmbeddingMode == "word2vec":
      self.word2idx, self.idx2vec = get_glove_vector(word2vecdict_path)  # word2vec词典
      diclen = len(self.word2idx.keys()) + 1
    elif EmbeddingMode == "glove":
      self.word2idx, self.idx2vec = get_glove_vector(glovedict_path)  # glove词典
      diclen = len(self.word2idx.keys()) + 1
    elif EmbeddingMode == "novec":  # 无需预训练词向量
      self.word2idx, self.idx2vec = get_glove_vector(glovedict_path)  # word2vec词典
      diclen = len(self.word2idx.keys()) + 1
    if EmbeddingMode == "word2vec" or EmbeddingMode == "glove":
      self.model = build_lstm(self.CateNum, diclen, self.idx2vec, mode="dict")
    elif EmbeddingMode == "novec":
      self.model = build_lstm(self.CateNum, diclen, self.idx2vec, mode="emb")


  def train_model(self):
    traindata = zip(self.x_train, self.y_train)
    testdata = zip(self.x_test, self.y_test)
    # print(list(traindata))  # [(string, onehot array)]
    x_train = sequence.pad_sequences(text2idx(self.word2idx, self.x_train), maxlen=maxlen)
    x_test = sequence.pad_sequences(text2idx(self.word2idx, self.x_test), maxlen=maxlen)
    y_train = np.array(self.y_train)
    y_test = np.array(self.y_test)

    if have_old_model:  # 加载旧模型
     self.model.load_weights(oldmodel_path)
    early_stopping = EarlyStopping(monitor='val_acc', patience=3)  # 早停法，防止过拟合
    plateau = ReduceLROnPlateau(monitor="val_acc", verbose=1, mode='max', factor=0.5,
                                patience=2)  # 当评价指标不再提升时，减少学习率
    model_path = LSTM_model_path + self.dataname + "_" + str(self.CateNum) + ".hdf5"
    checkpoint = ModelCheckpoint(model_path, monitor='val_acc',
                                 verbose=2, save_best_only=True, mode='max', save_weights_only=True)  # 保存最好的模型，增加时间信息
    self.model.fit(x_train, y_train, batch_size=lstm_batchsize, epochs=5, validation_data=(x_test, y_test),
              callbacks=[early_stopping, plateau, checkpoint])
    # 用测试数据评测模型
    print("训练好的LSTM在测试集上的表现：")
    y_pred = self.model.predict(x_test, batch_size=lstm_batchsize, verbose=1)
    y_test = [np.argmax(y) for y in y_test]
    y_pred = [np.argmax(y) for y in y_pred]
    accuracy, precision, recall, f1 = print_test_report(y_test, y_pred)

  def DoEvaluate(self, x_test, y_test, model_path):
    if x_test is None:  # 用数据集的测试集填充
      x_test = self.x_test
      y_test = self.y_test
    self.model.load_weights(model_path)  # 加载模型
    x_test = sequence.pad_sequences(text2idx(self.word2idx, x_test), maxlen=maxlen)
    y_test = np.array(y_test)
    y_test = [np.argmax(y) for y in y_test]
    y_pred = self.model.predict(x_test, batch_size=lstm_batchsize, verbose=1)
    # print(y_pred)
    y_pred = [np.argmax(y) for y in y_pred]
    accuracy, precision, recall, f1 = print_test_report(y_test, y_pred)


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
    self.model = None

  def train_model(self):
    self.model = build_bert(self.CateNum)
    traindata = list(zip(self.x_train, self.y_train))
    testdata = list(zip(self.x_test, self.y_test))
    test_true = [np.argmax(y) for x, y in testdata]
    test = data_generator(testdata, shuffle=False)
    early_stopping = EarlyStopping(monitor='val_acc', patience=3)  # 早停法，防止过拟合
    plateau = ReduceLROnPlateau(monitor="val_acc", verbose=1, mode='max', factor=0.5,
                                patience=2)  # 当评价指标不再提升时，减少学习率
    model_path = BERT_model_path + self.dataname + "_" + str(self.CateNum) + ".hdf5"
    checkpoint = ModelCheckpoint(model_path,
                                 monitor='val_acc', verbose=2,
                                 save_best_only=True, mode='max', save_weights_only=True)  # 保存最好的模型，增加时间信息
    # 将模型结构保存到json文件中
    json_string = self.model.to_json()
    structure_path = BERT_model_path + self.dataname + "_" + str(self.CateNum) + ".json"
    open(structure_path, "w").write(json_string)
    train = data_generator(traindata, shuffle=True)
    valid = data_generator(testdata, shuffle=True)
    # model.load_weights(BERT_model_path)  # 加载训练好的模型参数
    self.model.fit_generator(
      train.__iter__(),
      steps_per_epoch=len(train),
      epochs=1,
      validation_data=valid.__iter__(),
      validation_steps=len(valid),
      callbacks=[early_stopping, plateau] # , checkpoint]
    )
    self.model.save_weights(model_path)
    test_model_pred = np.zeros((len(testdata), self.CateNum))
    test_model_pred += self.model.predict_generator(test.__iter__(), steps=len(test), verbose=1)
    test_pred = [np.argmax(label) for label in test_model_pred]
    # 评测模型性能
    accuracy, precision, recall, f1 = print_test_report(test_true, test_pred)

  def DoEvaluate(self, x_test, y_test, model_path):
    structure_path = model_path.replace(".hdf5", ".json")  # 模型结构的存储路径
    # 从json文件中读取模型结构
    model_json = open(structure_path, "r").read()
    self.model = model_from_json(model_json, custom_objects=get_custom_objects())
    if x_test is None:
      x_test = self.x_test
      y_test = self.y_test
    testdata = list(zip(x_test, y_test))
    test = data_generator(testdata, shuffle=True)
    test_true = [np.argmax(y) for x, y in testdata]
    self.model.load_weights(model_path)
    test_model_pred = np.zeros((len(testdata), self.CateNum))
    test_model_pred += self.model.predict_generator(test.__iter__(), steps=len(test), verbose=1)
    test_pred = [np.argmax(label) for label in test_model_pred]
    idx = 0
    # for x in x_test:
      # print(x, test_true[idx], test_pred[idx])
      # idx += 1
    accuracy, precision, recall, f1 = print_test_report(test_true, test_pred)


def build_lstm(n_class, diclen, idx2vec, mode="dict"):
  # build the net
  model = Sequential()
  if mode == "dict":
    # 默认使用固定的词向量
    model.add(Embedding(output_dim=lstm_dim,
                        input_dim=diclen,
                        mask_zero=True,
                        weights=[idx2vec],
                        input_length=maxlen,
                        trainable=False,  # embedding层是否可训练
                        ))
  elif mode == "emb": # 直接训练词向量
    model.add(Embedding(len(idx2vec) + 1, lstm_dim, input_length=maxlen))  # 直接训练词向量
  model.add(Bidirectional(LSTM(units=maxlen)))
                               # output_dim=maxlen,
                               # activation='sigmoid',
                               # inner_activation='hard_sigmoid')))
  model.add(Dropout(0.5))
  model.add(Dense(n_class))
  model.add(Activation('softmax'))
  model.compile(loss='categorical_crossentropy',
                optimizer=Adam(1e-3),
                # optimizer="adam",
                metrics=['accuracy'])
  # print(model.summary())
  return model


# 大部分功能已被BiLSTM_classifier替代，后续可删去
def train_LSTM(traindata, testdata, rate):
    word2idx, idx2vec = get_glove_vector()
    print("加载词向量信息成功。")
    diclen = len(word2idx.keys()) + 1

    trainset = LSTMdata(list(traindata))
    trainset.get_sentences()
    x_train, x_test, y_train, y_test = train_test_split(trainset.sentences, trainset.labels,
                                                        test_size=(1-rate), random_state=127, stratify=trainset.labels)
    x_train = sequence.pad_sequences(text2idx(word2idx, x_train), maxlen=maxlen)
    x_test = sequence.pad_sequences(text2idx(word2idx, x_test), maxlen=maxlen)
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    # build the net
    model = build_lstm(3, diclen, idx2vec)

    if have_old_model:
     model.load_weights(oldmodel_path)
    early_stopping = EarlyStopping(monitor='val_acc', patience=3)  # 早停法，防止过拟合
    plateau = ReduceLROnPlateau(monitor="val_acc", verbose=1, mode='max', factor=0.5,
                                patience=2)  # 当评价指标不再提升时，减少学习率
    checkpoint = ModelCheckpoint('./model/' + date + '_conti_sentiment_analysis_lstm.hdf5', monitor='val_acc',
                                 verbose=2, save_best_only=True, mode='max', save_weights_only=True)  # 保存最好的模型，增加时间信息
    model.fit(x_train, y_train, batch_size=lstm_batchsize, epochs=20, validation_data=(x_test, y_test),
              callbacks=[early_stopping, plateau, checkpoint])

    eva = LSTMdata(list(testdata))
    eva.get_sentences()
    x_eva = sequence.pad_sequences(text2idx(word2idx, eva.sentences), maxlen=maxlen)
    y_eva = np.array(eva.labels)
    score, acc = model.evaluate(x_eva, y_eva, batch_size=lstm_batchsize)
    print("Score: ", score)
    print("Acc: ", acc)

def train_5cateBERT(model, traindata, validdata):
  early_stopping = EarlyStopping(monitor='val_acc', patience=3)  # 早停法，防止过拟合
  plateau = ReduceLROnPlateau(monitor="val_acc", verbose=1, mode='max', factor=0.5,
                              patience=2)  # 当评价指标不再提升时，减少学习率
  checkpoint = ModelCheckpoint('./bert_dump/sentiment_analysis_5cateBERT_230116.hdf5',
                               monitor='val_acc', verbose=2,
                               save_best_only=True, mode='max', save_weights_only=True)  # 保存最好的模型，增加时间信息
  train = data_generator(traindata, shuffle=True)
  valid = data_generator(validdata, shuffle=True)
  # model.load_weights("./bert_dump/sentiment_analysis_5cateBERT_230116.hdf5")  # 加载训练好的模型参数
  model.fit_generator(
    train.__iter__(),
    steps_per_epoch=len(train),
    epochs=5,
    validation_data=valid.__iter__(),
    validation_steps=len(valid),
    callbacks=[early_stopping, plateau, checkpoint]
  )
  print('./bert_dump/sentiment_analysis_5cateBERT_230116.hdf5, model saved successfully......')

def main():
    # build_bert(3)
    dataset, negset, posset, neuset, maxlen = get_labeled_data()
    trainset, testset = get_train_test_set(negset, posset, neuset)
    test_true = [np.argmax(y) for x, y in testset]
    print("训练集大小是{0}，测试集大小是{1}。".format(len(trainset), len(testset)))
    nfold = 5
    train_model_pred, test_model_pred = run_cv(nfold, trainset, testset)  # 训练基于BERT的情感分类模型

    # train_LSTM(trainset, testset, 0.9)  # 基于LSTM训练的情感分类模型

    # 直接加载模型
    test_model_pred = np.zeros((len(testset), 3))
    test = data_generator(testset, shuffle=True)
    model = build_bert(3)
    # model.load_weights("./bert_dump/sentiment_analysis_BERT_test0.hdf5")
    test_model_pred += model.predict_generator(test.__iter__(), steps=len(test), verbose=1)

    # 评测模型性能
    test_pred = [np.argmax(x) for x in test_model_pred]

    accuracy = accuracy_score(test_true, test_pred)
    precision = precision_score(test_true, test_pred, average="macro")
    recall = recall_score(test_true, test_pred, average="macro")
    f1 = f1_score(test_true, test_pred, average="macro")
    print("准确率{}，精确率{}，召回率{}，f1{}。".format(accuracy, precision, recall, f1))
    # print(test_pred)


def do_train_BiLSTM():  # 训练基于BiLSTM的情感分类模型
  dataset, negset, posset, neuset, maxlen = get_labeled_data()
  trainset, testset = get_train_test_set(negset, posset, neuset)
  train_LSTM(trainset, testset, 0.9)

def do_train_5cateBERT():
  # 五分类情感分析
  TrainData, DevData, TestData = get_labeled_data_5cate()
  print("训练集大小是{}，验证集大小是{}，验证集大小是{}。".format(len(TrainData), len(DevData), len(TestData)))
  test_true = [np.argmax(y) for x, y in TestData]
  model = build_bert(5)
  test = data_generator(TestData, shuffle=False)
  train_5cateBERT(model, TrainData, DevData)
  model_path = "./bert_dump/sentiment_analysis_5cateBERT_230116.hdf5"
  model.load_weights(model_path)
  test_model_pred = np.zeros((len(TestData), 5))
  test_model_pred += model.predict_generator(test.__iter__(), step=len(test), verbose=1)
  test_pred = [np.argmax(label) for label in test_model_pred]
  # 评测模型性能
  accuracy = accuracy_score(test_true, test_pred)
  precision = precision_score(test_true, test_pred, average="macro")
  recall = recall_score(test_true, test_pred, average="macro")
  f1 = f1_score(test_true, test_pred, average="macro")
  print("准确率{}，精确率{}，召回率{}，f1{}。".format(accuracy, precision, recall, f1))

if __name__ == "__main__":
    # print(tokenizer.encode(first="I like yyyyyxxxx."))
    # do_train_BiLSTM()
    # do_train_5cateBERT()
    # model = Word2Vec.load(root + '/svm_data/w2v_300d_model.pkl')  # 直接加载
    # word2idx, idx2vec = get_word2vec_vector(300, model)
    main()


