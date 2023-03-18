# 训练情感分类模型（use keras)
# ref: https://blog.csdn.net/asialee_bird/article/details/102747435
# 1007 样本maxlen调整（限制单词数maxlen=256）；选择wwm_uncased_L-24_H-1024_A-16（BERT-Large）作为base model进行训练。
import pickle
import random
from collections import defaultdict
import numpy as np
from gensim.corpora import Dictionary
from gensim.models import Word2Vec
from keras import Input, Model, Sequential
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.layers import Lambda, Dense, K, Embedding, LSTM, Dropout, Activation, Bidirectional
from keras.metrics import sparse_categorical_accuracy, categorical_accuracy
from keras.optimizers import Adam
from keras.preprocessing import sequence
from keras_bert import load_trained_model_from_checkpoint, Tokenizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold, train_test_split
from generate_corpus import get_labeled_data

# config_path = r"D:\Workspace\workspace\BERT_solution\uncased_L-12_H-768_A-12\bert_config.json"
# checkpoint_path = r"D:\Workspace\workspace\BERT_solution\uncased_L-12_H-768_A-12\bert_model.ckpt"
# dict_path = r"D:\Workspace\workspace\BERT_solution\uncased_L-12_H-768_A-12\vocab.txt"

# 使用BERT-Large, Uncased (Whole Word Masking)
config_path = "mnt/bert_pretrained_model/bert_config.json"
checkpoint_path = "mnt/bert_pretrained_model/bert_model.ckpt"
dict_path = "mnt/bert_pretrained_model/vocab.txt"

glovedict_path = "dict/glove.6B.300d.txt"  # 用于LSTM embedding
maxlen = 256
lstm_dim = 300  # 用glove 300d向量
lstm_batchsize = 32


def seq_padding(X, padding=0):  # 让每条文本的长度相同，用0填充
    L = [len(x) for x in X]
    ML = max(L)
    return np.array([
        np.concatenate([x, [padding] * (ML - len(x))]) if len(x) < ML else x for x in X
    ])


class data_generator:
    def __init__(self, data, batch_size=16, shuffle=True):
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
        checkpoint = ModelCheckpoint('mnt/bert_dump/' + 'sentiment_analysis_1007_uselarge_v2_' + str(i) + '.hdf5', monitor='val_acc', verbose=2,
                                     save_best_only=True, mode='max', save_weights_only=True)  # 保存最好的模型，增加时间信息
        train = data_generator(X_train, shuffle=True)
        valid = data_generator(X_valid, shuffle=True)
        test = data_generator(testdata, shuffle=False)
        # model.load_weights("mnt/bert_dump/sentiment_analysis_1007_uselarge_v1_0.hdf5")  # 加载训练好的模型参数
        model.fit_generator(
            train.__iter__(),
            steps_per_epoch=len(train),
            epochs=20,
            validation_data=valid.__iter__(),
            validation_steps=len(valid),
            callbacks=[early_stopping, plateau, checkpoint]
        )
        train_model_pred[test_fold, :] = model.predict_generator(valid.__iter__(), steps=len(valid), verbose=1)
        test_model_pred += model.predict_generator(test.__iter__(), steps=len(test), verbose=1)
        test_pred = [np.argmax(x) for x in test_model_pred]

        print('mnt/bert_dump/' + 'sentiment_analysis_1007_uselarge_v2_' + str(i) + '.hdf5')
        print(accuracy_score(test_true, test_pred))
        # del model
        # gc.collect()  # 清理内存
        K.clear_session()  # clear_session就是清除一个session
        # break
    return train_model_pred, test_model_pred


def get_glove_vector():
    f = open(glovedict_path, "r", encoding="utf8")
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
    model = Sequential()
    model.add(Embedding(output_dim=lstm_dim,
                        input_dim=diclen,
                        mask_zero=True,
                        weights=[idx2vec],
                        input_length=maxlen))
    model.add(LSTM(output_dim=maxlen,
                   activation='sigmoid',
                   inner_activation='hard_sigmoid'))
    model.add(Dropout(0.5))
    model.add(Dense(3))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(1e-5),
                  # optimizer="adam",
                  metrics=['accuracy'])
    print(model.summary())
    model.load_weights("model/sentiment_analysis_lstm.hdf5")
    early_stopping = EarlyStopping(monitor='val_acc', patience=3)  # 早停法，防止过拟合
    plateau = ReduceLROnPlateau(monitor="val_acc", verbose=1, mode='max', factor=0.5,
                                patience=2)  # 当评价指标不再提升时，减少学习率
    checkpoint = ModelCheckpoint('./model/' + 'sentiment_analysis_lstm_v2.hdf5', monitor='val_acc',
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
    # test_model_pred = np.zeros((len(testset), 3))
    # test = data_generator(testset, shuffle=False)
    # model = build_bert(3)
    # model.load_weights("./bert_dump/sentiment_analysis_1005_v3_0.hdf5")
    # test_model_pred += model.predict_generator(test.__iter__(), steps=len(test), verbose=1)

    # 评测模型性能
    test_pred = [np.argmax(x) for x in test_model_pred]
    print(accuracy_score(test_true, test_pred))
    # print(test_pred)


if __name__ == "__main__":
    token_dict = get_tokendict()
    tokenizer = Tokenizer(token_dict)
    # print(tokenizer.encode(first="I like yyyyyxxxx."))
    main()
