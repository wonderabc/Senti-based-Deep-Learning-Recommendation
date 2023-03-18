# 计算user、item的主题情感特征
# 0421 使用TF-IDF法计算主题情感特征
# 计算software、magazine数据集的主题情感特征
import ast
import json
from collections import defaultdict
from math import log
import demjson
import numpy as np
from sklearn import preprocessing
from tqdm import tqdm

mode = "software"
mode = "magazine"
if mode == "software":
  topic_num = 10
elif mode == "magazine":
  topic_num = 12

review_sentiment_path = "./data/0504_" + mode + "_review_topic_sentiment.txt"
# users_save_path = "./data/0102_users_topic_feature.txt"
# products_save_path = "./data/0102_products_topic_feature.txt"
users_save_path_new = "./data/0505_" + mode + "_users_topic_feature_tfidf.txt"
products_save_path_new = "./data/0505_" + mode + "_products_topic_feature_tfidf.txt"

# record_path = "../data/reviews_include_gameinfo/1014_reviews_include_gameinfo_100w_0_100w.txt"  # 包含productid信息
# include_product_save_path = "./data/1128_preprocessed_generated_include_product.txt"
# include_topic_feature_save_path = "./data/1128_preprocessed_generated_include_topic_feature.txt"


def get_tfidf_topic_feature(id2reviews, id2topic_features):  # tf-idf法计算topic feature
  min_max_scaler = preprocessing.MinMaxScaler()  # min_max 归一化
  id_len = len(id2reviews.keys())
  have_topic = [0] * (topic_num + 1)
  for i in range(1, topic_num + 1):
    for id in id2reviews.keys():
      if id2topic_features[id][i][0] > 0:
        have_topic[i] += 1
  tf_idf = defaultdict(lambda: np.zeros((topic_num + 1, 1)))
  for id in id2reviews.keys():
    topic_feature = id2topic_features[id]
    tot = sum(topic_feature)  # 所有主题总频次
    if tot > 0:  # 没有主题特征
      for topic in range(1, topic_num + 1):
        if topic_feature[topic][0] == 0:
          continue
        tf_idf[id][topic][0] = topic_feature[topic][0] / tot * log(id_len / have_topic[topic])
    tf_idf[id] = min_max_scaler.fit_transform(tf_idf[id]).T
    # print(id, tf_idf[id])
  return tf_idf

def get_user_item_topic_feature():  # 获得user / item 的主题情感特征
    product2reviews_dic = defaultdict(list)  # 存储product对应的review列表
    userid2reviews_dic = defaultdict(list)  # 存储userid对应的review列表
    f_reviews = open(review_sentiment_path, "r", encoding="utf8")
    lines = f_reviews.readlines()
    form_err = 0  # 格式错误计数

    # 获取用户评价集、产品被评价集
    for i in tqdm(range(len(lines))):
      line = lines[i].strip()
      # reviewerName + "\t" + reviewerID + "\t" + asin + "\t" + text + "\t" + str(list(topic_sentiment_info[0]))
      parts = line.split("\t")
      if len(parts) != 5:
        form_err += 1
        continue
      reviewerName, user_id, product_id, text, topic_sentiment_info = parts
      user_id = user_id.strip()
      product_id = product_id.strip()
      topic_sentiment_info = ast.literal_eval(topic_sentiment_info)  # 存储为list
      # print(type(topic_sentiment_info))
      userid2reviews_dic[user_id].append(topic_sentiment_info)
      product2reviews_dic[product_id].append(topic_sentiment_info)

    print("格式错误的有{0}条。".format(form_err))
    user_len = len(list(userid2reviews_dic.keys()))  # 用户数
    product_len = len(list(product2reviews_dic.keys()))  # 产品数
    print("共有{0}名用户，{1}个产品。".format(user_len, product_len))

    min_max_scaler = preprocessing.MinMaxScaler()  # min_max 归一化
    users_topic_fea = defaultdict(lambda: np.zeros((topic_num + 1, 1)))
    products_topic_fea = defaultdict(lambda: np.zeros((topic_num + 1, 1)))
    # 计算用户主题情感特征
    for userid in list(userid2reviews_dic.keys()):
      reviews = userid2reviews_dic[userid]
      user_topic_feature = np.zeros((topic_num + 1, 1))
      for review in reviews:
        # print(review)
        # review = list(review)
        for i in range(len(review)):
          val = float(review[i])
          user_topic_feature[i] += val
      users_topic_fea[userid] = user_topic_feature
    users_topic_fea = get_tfidf_topic_feature(userid2reviews_dic, users_topic_fea)

    # 计算产品主题情感特征
    for productid in list(product2reviews_dic.keys()):
      reviews = product2reviews_dic[productid]
      product_topic_fea = np.zeros((topic_num + 1, 1))
      for review in reviews:
        # review = list(review)
        for i in range(len(review)):
          val = float(review[i])
          product_topic_fea[i] += val
      products_topic_fea[productid] = product_topic_fea
      # print(products_topic_fea[productid])
    products_topic_fea = get_tfidf_topic_feature(product2reviews_dic, products_topic_fea)

    # 将主题情感特征存储为文件
    # f_save_users = open(users_save_path, "w", encoding="utf8")
    # f_save_products = open(products_save_path, "w", encoding="utf8")
    f_save_users = open(users_save_path_new, "w", encoding="utf8")
    f_save_products = open(products_save_path_new, "w", encoding="utf8")
    for userid in list(users_topic_fea.keys()):
      tmp = str(userid) + "\t" + str(list(users_topic_fea[userid][0])) + "\n"
      f_save_users.write(tmp)
    for productid in list(products_topic_fea.keys()):
      tmp = str(productid) + "\t" + str(list(products_topic_fea[productid][0])) + "\n"
      f_save_products.write(tmp)


def main():
    get_user_item_topic_feature()
    # get_record_feature()
    # generate_trainset()


if __name__ == "__main__":
    main()
