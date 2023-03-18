# 更多指标评估
from collections import defaultdict
import joblib
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from FM_solution.get_inter_data import get_data
from dataset.get_data import get_data_from_file
from evaluation_util.Indicator import Evaluator
from CF.main_CF import *

top_k_list = [3, 5, 10, 20]
seed = 127
# datasource = ["amazon"]
datasource = ["amazon", "software", "magazine"]
# featuremode = ["normal"]
featuremode = ["normal", "normal", "normal"]
# modelpath_list = ["./model/best_model_amazon"]
if __name__ == "__main__":
  for top_k in top_k_list:
    for dsource, fmode in zip(datasource, featuremode):
      data_path = ""
      if dsource == "amazon":
        data_path = "../FM_solution/data/amazon_u.data"
      elif dsource == "software":
        data_path = "../FM_solution/data/software_u.data"
      elif dsource == "magazine":
        data_path = "../FM_solution/data/magazine_u.data"

      data, u_trans_dic, i_trans_dic = get_data(data_path)
      u_id2name_dic = defaultdict(lambda: "")
      i_id2name_dic = defaultdict(lambda: "")
      for key in u_trans_dic:
        u_id2name_dic[u_trans_dic[key]] = key
      for key in i_trans_dic:
        i_id2name_dic[i_trans_dic[key]] = key
      data_group = []  # 三元组的形式存储打分
      for i in range(len(data)):
        data_group.append((data.loc[i, 0], data.loc[i, 1], data.loc[i, 2]))

      u_max_id = max(data[0]) + 1
      i_max_id = max(data[1]) + 1
      print("u_max_id: {}, i_max_id: {}。".format(u_max_id, i_max_id))

      model = CF(u_id2name_dic, i_id2name_dic, data_group, mode=dsource, k=5, n=top_k)
      x_test = []
      y_test = []
      for key in u_id2name_dic:
        model.recommendByUser(key)  # 生成推荐列表
        recommendList = model.recommandList
        labellist = model.userDict[key]
        for s, productid in recommendList:
          for pid, label in labellist:
            if pid == productid:  # label数据中包含
              x_test.append([key, productid, s])
              y_test.append(label)
              break
      assert len(x_test) == len(y_test)
      evaluate_task = Evaluator(x_test, y_test, model, top_k=top_k, data_source=dsource, model_name="CF", u_trans_dic=u_trans_dic, i_trans_dic=i_trans_dic)
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
      print("model: CF, top_k: {}, datasource: {}, featuremode: {}，评估结果：".format(top_k, dsource, fmode))
      print("HR: {:.4f}, MRR: {:.4f}, ILS: {:.4f}, NDCG: {:.4f}.".format(evaluate_task.HR, evaluate_task.MRR, evaluate_task.ILS,
                                                       evaluate_task.ndcg))
