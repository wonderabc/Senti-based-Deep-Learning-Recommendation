# 抽取amazon数据集产品描述的特征
import string

from gensim import corpora
from tqdm import tqdm

data_path = "../../data/amazon/records_include_gameanduser_info_0109.txt"
stopword_path = "../stopwords.txt"
save_path = "./model/products_topics_from_words.txt"  # 词频统计结果


def clean(text):
  text = text.replace(";", " ").replace(",", " ").replace(".", " ").replace(":", " ")
  stop_free = " ".join([i for i in text.split(' ') if i not in stopword_list])
  punc_free = " ".join([ch for ch in stop_free.split(' ') if ch not in exclude])  # 删掉标点
  return punc_free


def main():
  f = open(data_path, "r", encoding="utf8")
  lines = f.readlines()
  doc_clean = []
  for line in lines:
    parts = line.strip().split("\t")
    if len(parts) != 17:
      continue
    reviewerID, asin, reviewerName, helpful, reviewText, overall, summary, unixReviewTime, \
      reviewTime, product_description, product_price, product_related, product_salesRank, \
      product_categories, product_title, product_brand, user_categories = parts
    product_description = product_description.strip()
    doc_clean.append(clean(product_description).split())

  print("\n生成词典……")
  dictionary = corpora.Dictionary(doc_clean)
  dictionary.save('./model/product_description.dict')
  print("生成词典结束。")

  dictionary = corpora.Dictionary.load('./model/product_description.dict')  # 加载词典
  frequency = dictionary.dfs
  frequencylist = []
  for id in frequency.keys():
    frequencylist.append((dictionary[id], frequency[id]))
  frequencylist = sorted(frequencylist, key=lambda t: t[1], reverse=True)  # 降序排列
  f = open(save_path, "w", encoding="utf8")
  for i in tqdm(range(len(frequencylist))):
    item = frequencylist[i]
    line = str(item[0]).strip() + "\t" + str(item[1]) + "\n"
    f.write(line)
  f.close()


def get_stopwords():
  stopwords = []
  f = open(stopword_path, "r", encoding="utf8")
  lines = f.readlines()
  for line in lines:
    stopwords.append(line.strip())
  return stopwords


if __name__ == "__main__":
  stopword_list = get_stopwords()
  exclude = set(string.punctuation)  # 标点符号
  main()
