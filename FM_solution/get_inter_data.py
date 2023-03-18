# 获得交互数据
# steam / amazon
import pandas as pd
from collections import defaultdict

def get_data(path):  # 获取数据并转换用户、商品idx
  data = pd.read_csv(path, sep="\t", header=None)
  data[0] = data[0].astype(str)
  data[1] = data[1].astype(str)
  data[2] = data[2].astype(float)
  # u_idx, i_idx 映射
  u_set, i_set = defaultdict(int), defaultdict(int)
  u_trans_dic, i_trans_dic = defaultdict(int), defaultdict(int)
  u_now_idx, i_now_idx = 1, 1
  data_len = len(data)
  for i in range(data_len):
    if u_trans_dic[data.loc[i, 0]] == 0:  # 该idx没有出现过
      u_trans_dic[data.loc[i, 0]] = u_now_idx
      data.loc[i, 0] = u_now_idx
      u_set[u_now_idx] += 1
      u_now_idx += 1
    else:
      data.loc[i, 0] = u_trans_dic[data.loc[i, 0]]
      u_set[data.loc[i, 0]] += 1
    if i_trans_dic[data.loc[i, 1]] == 0:
      i_trans_dic[data.loc[i, 1]] = i_now_idx
      data.loc[i, 1] = i_now_idx
      i_set[i_now_idx] += 1
      i_now_idx += 1
    else:
      data.loc[i, 1] = i_trans_dic[data.loc[i, 1]]
      i_set[data.loc[i, 1]] += 1
  # print(len(u_set), len(i_set))
  return data, u_trans_dic, i_trans_dic


def get_data_amazon():
  data = pd.read_csv("./data/amazon_u.data", sep="\t", header=None)
  data[0] = data[0].astype(str)
  data[1] = data[1].astype(str)
  data[2] = data[2].astype(float)
  # u_idx, i_idx 映射
  u_set, i_set = defaultdict(int), defaultdict(int)
  u_trans_dic, i_trans_dic = defaultdict(int), defaultdict(int)
  u_now_idx, i_now_idx = 1, 1
  data_len = len(data)
  for i in range(data_len):
    if u_trans_dic[data.loc[i, 0]] == 0:  # 该idx没有出现过
      u_trans_dic[data.loc[i, 0]] = u_now_idx
      data.loc[i, 0] = u_now_idx
      u_set[u_now_idx] += 1
      u_now_idx += 1
    else:
      data.loc[i, 0] = u_trans_dic[data.loc[i, 0]]
      u_set[data.loc[i, 0]] += 1
    if i_trans_dic[data.loc[i, 1]] == 0:
      i_trans_dic[data.loc[i, 1]] = i_now_idx
      data.loc[i, 1] = i_now_idx
      i_set[i_now_idx] += 1
      i_now_idx += 1
    else:
      data.loc[i, 1] = i_trans_dic[data.loc[i, 1]]
      i_set[data.loc[i, 1]] += 1
  # print(len(u_set), len(i_set))
  return data, u_trans_dic, i_trans_dic

if __name__ == "__main__":
  data = get_data('./data/software_u.data')