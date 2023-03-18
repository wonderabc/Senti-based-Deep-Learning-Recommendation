# 更多指标评估
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from dataset.get_data import get_data_from_file
from evaluation_util.Indicator import Evaluator
top_k_list = [3, 5, 10, 20]
seed = 127
# datasource = ["amazon", "amazon"]
datasource = ["software", "software", "magazine", "magazine"]
featuremode = ["normal", "with-sentiment","normal", "with-sentiment"]
modelpath_list = ["./model/0509_software_normal_lr.model", "./model/0509_software_with-sentiment_lr.model",
                  "./model/0517_magazine_normal_lr.model", "./model/0517_magazine_with-sentiment_lr.model"]

if __name__ == "__main__":
  # if datasource == "steam":
    # data, sparse_features, continuous_features, col_names = get_data_from_file(mode=featuremode, datasource=datasource)
  # else:
  for top_k in top_k_list:
    for dsource, fmode, modelpath in zip(datasource, featuremode, modelpath_list):
      data, sparse_features, continuous_features, col_names, scoreset = get_data_from_file(mode=fmode, datasource=dsource, datamode="evaluate")

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
      data["score"] = scoreset
      train, test = train_test_split(data, test_size=0.1, random_state=seed)  # 随机获取测试集
      y_test = test['label']
      x_test = test.drop(columns=['label'])
      labelset = ['score', 'userid', 'productid']
      y_test_userid = test[labelset]
      x_test = x_test.drop(columns=labelset)
      model = joblib.load(modelpath)
      evaluate_task = Evaluator(x_test, y_test_userid, model, top_k=top_k, data_source=dsource, model_name="LR")
      print("生成推荐列表。")
      evaluate_task.cal_recommendlist()
      print("计算HR。")
      evaluate_task.cal_HR()
      print("计算MRR。")
      evaluate_task.cal_MRR()
      print("计算ILS。")
      evaluate_task.cal_ILS()
      print("计算NDCG。")
      evaluate_task.cal_NDCG()
      print("top_k: {}, datasource: {}, featuremode: {}，评估结果：".format(top_k, dsource, fmode))
      print("HR: {:.4f}, MRR: {:.4f}, ILS: {:.4f}, NDCG: {:.4f}.".format(evaluate_task.HR, evaluate_task.MRR, evaluate_task.ILS,
                                                       evaluate_task.ndcg))
