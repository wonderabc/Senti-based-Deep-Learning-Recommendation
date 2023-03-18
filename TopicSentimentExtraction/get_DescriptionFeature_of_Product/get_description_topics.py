import string
import gensim
from gensim import corpora
from tqdm import tqdm
source_path = "../data/amazon/records_include_gameanduser_info_0109.txt"
word_fre_save_path = "./data/topics_from_words_after_filtering.txt"
date = "0420"

def clean(text):
    stop_free = " ".join([i for i in text.split(' ') if i not in stopword])
    punc_free = " ".join([ch for ch in stop_free.split(' ') if ch not in exclude])  # 删掉标点
    return punc_free


def main():
    doclist = []  # 文档集合
    f = open(source_path, "r", encoding="utf8")
    lines = f.readlines()
    n = len(lines)
    for line in lines:
      parts = line.strip().split("\t")
      if len(parts) != 17:
        continue
      reviewerID, asin, reviewerName, helpful, reviewText, overall, summary, unixReviewTime, \
        reviewTime, product_description, product_price, product_related, product_salesRank, \
        product_categories, product_title, product_brand, user_categories = parts
      product_description = product_description.strip().replace(";", " ; ").replace(",", " , ").replace(".", " . ").replace(":", " : ").replace("/", " / ")
      doclist.append(product_description)

    print("\n开始数据清洗……")
    doc_clean = []
    for i in tqdm(range(len(doclist))):
      doc = doclist[i]
      doc_clean.append(clean(doc).split())

    print("\n生成词典……")
    dictionary = corpora.Dictionary(doc_clean)
    dictionary.save('./model/' + date + '_product_description.dict')
    print("生成词典结束。")

    # 加载词典
    dictionary = corpora.Dictionary.load('./model/' + date + '_product_description.dict')
    no_below = n * 0.001  # 过滤出现次数少于0.1%的
    no_above = 0.2  # 过滤出现频率高于20%的
    dictionary.filter_extremes(no_below=no_below, no_above=no_above, keep_n=10000)  # 过滤低频词和高频词，保留1w单词（实际上只有1000+单词）
    dictionary.save('./model/' + date + '_product_description_after_filtering.dict')

    f = open(word_fre_save_path, "w", encoding="utf8")
    frequency = dictionary.dfs
    # print(frequency[0])
    frequencylist = []
    for id in frequency.keys():
        frequencylist.append((dictionary[id], frequency[id]))
    frequencylist = sorted(frequencylist, key=lambda t: t[1], reverse=True)  # 降序排列
    for i in tqdm(range(len(frequencylist))):
      item = frequencylist[i]
      line = str(item[0]).strip() + "\t" + str(item[1]) + "\n"
      f.write(line)

if __name__ == "__main__":
  # get punctuation
  exclude = set(string.punctuation)  # 标点符号
  # get stopwords
  stopword = set()
  fr = open('./stopwords.txt', mode='r', encoding='utf-8')
  for word in fr:
    stopword.add(word.strip())
  main()