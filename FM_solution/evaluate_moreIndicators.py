# 更多指标评估
import joblib
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from FM_solution.get_inter_data import get_data
from dataset.get_data import get_data_from_file
from evaluation_util.Indicator import Evaluator
from FM_solution.FM import *
top_k_list = [3, 5, 10, 20]
seed = 127
# datasource = ["amazon"]
datasource = ["software", "magazine"]
# featuremode = ["normal"]
featuremode = ["normal", "normal"]
modelpath_list = ["./model/best_model_software_0509", "./model/best_model_magazine_0517"]

if __name__ == "__main__":
  # if datasource == "steam":
    # data, sparse_features, continuous_features, col_names = get_data_from_file(mode=featuremode, datasource=datasource)
  # else:
  for top_k in top_k_list:
    for dsource, fmode, modelpath in zip(datasource, featuremode, modelpath_list):
      data_path = ""
      if dsource == "amazon":
        data_path = "./data/amazon_u.data"
      elif dsource == "software":
        data_path = "./data/software_u.data"
      elif dsource == "magazine":
        data_path = "./data/magazine_u.data"
      data, u_trans_dic, i_trans_dic = get_data(data_path)
      u_max_id = max(data[0]) + 1
      i_max_id = max(data[1]) + 1
      print("u_max_id: {}, i_max_id: {}。".format(u_max_id, i_max_id))
      x, y = data.iloc[:, :2], data.iloc[:, 2]
      x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=seed)
      x_test = FmDataset(np.array(x_test[0]), np.array(x_test[1]), np.array(y_test).astype(np.float32))
      model = torch.load(modelpath)
      evaluate_task = Evaluator(x_test, y_test, model, top_k=top_k, data_source=dsource, model_name="FM", u_trans_dic=u_trans_dic, i_trans_dic=i_trans_dic)
      print("生成推荐列表。")
      evaluate_task.cal_recommendlist()
      # print(evaluate_task.userid_recommendlist)
      print("计算HR。")
      evaluate_task.cal_HR()
      print("计算MRR。")
      evaluate_task.cal_MRR()
      print("计算ILS。")
      evaluate_task.cal_ILS()
      print("计算NDCG。")
      evaluate_task.cal_NDCG()
      print("model: FM, top_k: {}, datasource: {}, featuremode: {}，评估结果：".format(top_k, dsource, fmode))
      print("HR: {:.4f}, MRR: {:.4f}, ILS: {:.4f}, NDCG: {:.4f}.".format(evaluate_task.HR, evaluate_task.MRR, evaluate_task.ILS,
                                                       evaluate_task.ndcg))
