# 评估BERT-base情感分类模型的性能
from keras.preprocessing import sequence
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sentiment_analysis.generate_corpus import get_labeled_data
from sentiment_analysis.train_sentiment_model import get_train_test_set, data_generator, build_bert_add_BiLSTM, \
  build_bert, get_glove_vector, build_lstm, LSTMdata, text2idx
import numpy as np
maxlen = 256
do_eval_bert = False
bert_classifier_path = "./bert_dump/sentiment_analysis_1005_v3_0.hdf5"

do_eval_bert_addbilstm = True
bert_addbilstm_path = "./bert_dump/sentiment_analysis_1006_addBiLSTM_v3_2.hdf5"

do_eval_LSTM = True
BiLSTM_path = "./model/0319_conti_sentiment_analysis_lstm.hdf5"

if __name__ == "__main__":
  if do_eval_bert:
    dataset, negset, posset, neuset, _ = get_labeled_data()
    trainset, testset = get_train_test_set(negset, posset, neuset)
    test_true = [np.argmax(y) for x, y in testset]
    test_model_pred = np.zeros((len(testset), 3))
    test = data_generator(testset, shuffle=False)

    # 加载模型
    model = build_bert(3)
    model.load_weights(bert_classifier_path)

    test_model_pred += model.predict_generator(test.__iter__(), steps=len(test), verbose=1)
    test_pred = [np.argmax(x) for x in test_model_pred]
    print("微调bert训练情感分类模型表现如下：")
    accuracy = accuracy_score(test_true, test_pred)
    precision = precision_score(test_true, test_pred, average="macro")
    recall = recall_score(test_true, test_pred, average="macro")
    f1 = f1_score(test_true, test_pred, average="macro")
    print("准确率{}，精确率{}，召回率{}，f1值{}。".format(accuracy, precision, recall, f1))

  if do_eval_bert_addbilstm:
    dataset, negset, posset, neuset, _ = get_labeled_data()
    trainset, testset = get_train_test_set(negset, posset, neuset)
    test_true = [np.argmax(y) for x, y in testset]
    # print("训练集大小是{0}，测试集大小是{1}。".format(len(trainset), len(testset)))
    test_model_pred = np.zeros((len(testset), 3))
    test = data_generator(testset, shuffle=False)

    # 加载模型
    model = build_bert_add_BiLSTM(3)
    model.load_weights(bert_addbilstm_path)

    test_model_pred += model.predict_generator(test.__iter__(), steps=len(test), verbose=1)
    test_pred = [np.argmax(x) for x in test_model_pred]
    print("bert基础上增加BiLSTM层的情感分类模型表现如下：")
    accuracy = accuracy_score(test_true, test_pred)
    precision = precision_score(test_true, test_pred, average="macro")
    recall = recall_score(test_true, test_pred, average="macro")
    f1 = f1_score(test_true, test_pred, average="macro")
    print("准确率{}，精确率{}，召回率{}，f1值{}。".format(accuracy, precision, recall, f1))

  if do_eval_LSTM:
    # 加载模型
    word2idx, idx2vec = get_glove_vector()
    print("加载词向量信息成功。")
    diclen = len(word2idx.keys()) + 1
    model = build_lstm(3, diclen, idx2vec)
    model.load_weights(BiLSTM_path)

    dataset, negset, posset, neuset, _ = get_labeled_data()
    trainset, testset = get_train_test_set(negset, posset, neuset)
    eva = LSTMdata(list(testset))
    eva.get_sentences()
    x_eva = sequence.pad_sequences(text2idx(word2idx, eva.sentences), maxlen=maxlen)
    y_eva = np.array(eva.labels)
    y_eva_label = [np.argmax(y) for y in list(y_eva)]
    y_pred = model.predict(x_eva, batch_size=32)
    y_pred_label = [np.argmax(y) for y in list(y_pred)]
    print("BiLSTM训练情感分类模型表现如下：")
    accuracy = accuracy_score(y_eva_label, y_pred_label)
    precision = precision_score(y_eva_label, y_pred_label, average="macro")
    recall = recall_score(y_eva_label, y_pred_label, average="macro")
    f1 = f1_score(y_eva_label, y_pred_label, average="macro")
    print("准确率{}，精确率{}，召回率{}，f1值{}。".format(accuracy, precision, recall, f1))
    # print(y_pred)


