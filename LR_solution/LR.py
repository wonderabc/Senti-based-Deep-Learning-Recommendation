from math import sqrt

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, accuracy_score, roc_auc_score, recall_score, mean_absolute_error, \
  mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from dataset.get_data import get_data_from_file
seed = 127
date = "0517"
datasource = "amazon"
datasource = "software"
datasource = "magazine"
featuremode = "normal"
# featuremode = "with-sentiment"


def train_lr_model():
  scoreset = None
  if datasource == "steam":
    data, sparse_features, continuous_features, col_names = get_data_from_file(mode=featuremode, datasource=datasource)
  else:
    data, sparse_features, continuous_features, col_names, scoreset = get_data_from_file(mode=featuremode, datasource=datasource)
  print("记录个数{}, 稀疏特征数{}, 连续特征数{}。".format(len(data), len(sparse_features), len(continuous_features)))
  data[sparse_features] = data[sparse_features].fillna('-1', )
  data[continuous_features] = data[continuous_features].fillna('0', )
  data["label"] = data["label"].astype("float")
  for feat in sparse_features:
    labelencoder = LabelEncoder()
    data[feat] = data[feat].astype("string")
    data[feat] = labelencoder.fit_transform(data[feat])
  minmaxscaler = MinMaxScaler(feature_range=(0, 1))
  for feat in continuous_features:
    data[feat] = data[feat].astype("float")
  data[continuous_features] = minmaxscaler.fit_transform(data[continuous_features])
  if scoreset is not None:
    data["score"] = scoreset
  train, test = train_test_split(data, test_size=0.1, random_state=seed)  # 随机获取测试集
  if scoreset is not None:
    y_train_score = train['score']
  y_train = train['label']
  x_train = train.drop(columns=['label', 'score'])
  if scoreset is not None:
    y_test_score = test['score']
  y_test = test['label']
  x_test = test.drop(columns=['label', 'score'])
  # x_train, x_test, y_train, y_test = train_test_split(all_data, all_target, test_size=0.1, random_state=127)
  print("开始训练LR模型……")
  lr = LogisticRegression(verbose=1, max_iter=500, tol=1e-3)
  lr.fit(x_train, y_train)
  print("LR模型训练结束。")
  joblib.dump(lr, "./model/" + date + "_" + datasource + "_" + featuremode + "_lr.model")
  tr_logloss = log_loss(y_train, lr.predict_proba(x_train)[:, 1])
  test_logloss = log_loss(y_test, lr.predict_proba(x_test)[:, 1])
  print('tr-logloss: ', tr_logloss)
  print('test-logloss: ', test_logloss)

  y_train = [int(y) for y in y_train]
  y_train_pred = list(lr.predict_proba(x_train)[:, 1])
  for i in range(len(y_train_pred)):
    y = y_train_pred[i]
    if y <= 0.5:
      y_train_pred[i] = 0
    else:
      y_train_pred[i] = 1
  tr_accuracy = accuracy_score(y_train, y_train_pred)

  y_test = [int(y) for y in y_test]
  y_test_pred = list(lr.predict_proba(x_test)[:, 1])
  y_test_pred_label = []
  for i in range(len(y_test_pred)):
    y = y_test_pred[i]
    if y <= 0.5:
      y_test_pred_label.append(0)
    else:
      y_test_pred_label.append(1)
  test_accuracy = accuracy_score(y_test, y_test_pred_label)
  auc_score = roc_auc_score(y_test, y_test_pred)
  recall = recall_score(y_test, y_test_pred_label, average="macro")

  # calculate mae and rmse
  mae, rmse = 0.0, 0.0
  if scoreset is not None:
    min_val, max_val = 1.0, 5.0
    y_test_score = [float(score) for score in y_test_score]
    y_test_score_pred = []
    for i in range(len(y_test_pred)):
      y_test_score_pred.append(min_val + y_test_pred[i] * (max_val - min_val))
    mae = mean_absolute_error(y_test_score, y_test_score_pred)
    rmse = sqrt(mean_squared_error(y_test_score, y_test_score_pred))
  # print("tr-accuracy: ", tr_accuracy)
  print("test-auc-score: {:.4f}, test-accuracy: {:.4f}, test-recall: {:.4f}".format(auc_score, test_accuracy, recall))
  if scoreset is not None:
    print("test-mae: {:.4f}, test-rmse: {:.4f}".format(mae, rmse))


if __name__ == "__main__":
  train_lr_model()