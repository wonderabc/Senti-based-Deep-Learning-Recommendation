# 获得评价主题
import string
import pandas as pd
from gensim import corpora
from tqdm import tqdm
from deal_dataset.make_data_into_csv import getDF
software_meta_columns = ['category', 'tech1', 'description', 'fit', 'title', 'also_buy', 'tech2', 'brand', 'feature', 'rank', 'also_view', 'main_cat', 'similar_item', 'date', 'price', 'asin', 'imageURL', 'imageURLHighRes', 'details']
software_review_columns = ['overall', 'verified', 'reviewTime', 'reviewerID', 'asin', 'style', 'reviewerName', 'reviewText', 'summary', 'unixReviewTime', 'vote', 'image']
magazine_meta_columns = ['category', 'tech1', 'description', 'fit', 'title', 'also_buy', 'tech2', 'brand', 'feature', 'rank', 'also_view', 'details', 'main_cat', 'similar_item', 'date', 'price', 'asin', 'imageURL', 'imageURLHighRes']
magazine_review_columns = ['overall', 'verified', 'reviewTime', 'reviewerID', 'asin', 'reviewerName', 'reviewText', 'summary', 'unixReviewTime', 'vote', 'style', 'image']

mode = "software"
mode = "magazine"
if mode == "software":
  meta_path = "./data/meta_Software.json.gz"
  review_path = "./data/Software_5.json.gz"
  dic_save_path = "./model/software_reviews.dict"
  topic_save_path = "./data/software_topics.txt"
elif mode == "magazine":
  meta_path = "./data/meta_Magazine_Subscriptions.json.gz"
  review_path = "./data/Magazine_Subscriptions_5.json.gz"
  dic_save_path = "./model/magazines_reviews.dict"
  topic_save_path = "./data/magazines_topics.txt"


def getmetas():
  # metalist = []
  # 直接从json文件中获得
  meta_DF = getDF(meta_path)  # DataFrame格式
  return meta_DF

def getreviews():
  review_DF = getDF(review_path)
  return review_DF

def clean(text):
  text = text.replace(",", " , ").replace(".", " . ").replace(";", " ; ").replace(":", " : ").replace('"', ' " ')
  stop_free = " ".join([i.strip() for i in text.split(' ') if i not in stopword and len(i.strip()) > 0])
  punc_free = " ".join([ch for ch in stop_free.split(' ') if ch not in exclude])  # 删掉标点
  return punc_free

def generate_topics():  # 生成主题频次信息
  reviews = getreviews()
  doclist = []
  for i in range(len(reviews)):
    doclist.append(str(reviews.loc[i, "reviewText"]))

  print("\n开始数据清洗……")
  doc_clean = []
  for i in tqdm(range(len(doclist))):
    doc = doclist[i]
    doc_clean.append(clean(doc).split())

  print("\n生成词典……")
  dictionary = corpora.Dictionary(doc_clean)
  n = len(doc_clean)
  no_below = n * 0.001  # 过滤出现次数少于0.1%的
  no_above = 0.2  # 过滤出现频率高于20%的
  dictionary.filter_extremes(no_below=no_below, no_above=no_above, keep_n=1000)  # 过滤低频词和高频词
  dictionary.save(dic_save_path)
  print("生成词典结束。")

  print("输出词频统计结果……")
  f = open(topic_save_path, encoding="utf8", mode="w")
  dictionary = corpora.Dictionary.load(dic_save_path)  # 加载dictionary
  frequency = dictionary.dfs
  frequencylist = []
  for id in frequency.keys():
    frequencylist.append((dictionary[id], frequency[id]))
  frequencylist = sorted(frequencylist, key=lambda t: t[1], reverse=True)  # 降序排列
  for i in tqdm(range(len(frequencylist))):
    item = frequencylist[i]
    line = str(item[0]).strip() + "\t" + str(item[1]) + "\n"
    f.write(line)

if __name__ == "__main__":
  stopword = set()
  fr = open('stopwords.txt', mode='r', encoding='utf-8')
  for word in fr:
    stopword.add(word.strip())
  exclude = set(string.punctuation)
  # generate_topics()
  metas = getmetas()
  print(list(metas.columns.values))
  reviews = getreviews()
  print(list(reviews.columns.values))