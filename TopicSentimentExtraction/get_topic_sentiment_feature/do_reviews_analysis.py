# 对review的分句做主题分类和情感分析
# 规定情感得分: positive: 1, neutral: 0.6, negative: 0.2 (1014方案)
import re
import numpy as np
from collections import defaultdict
from sklearn import preprocessing
from tqdm import tqdm

from util.do_sentiment_analysis import SentimentOfSentence
topicdictpath = "../get_features/data/topics.dic"  # 获取主题（词）列表
inputpath = "../data/reviews_include_gameinfo/1014_reviews_include_gameinfo_100w_0_100w.txt"
resultpath = "./1014_review_topic_sentiment.txt"  # 评论的主题-情感得分情况


def get_topics():  # 获取主题（词）列表，形如 {1:[story]}
    dic = defaultdict(list)
    f = open(topicdictpath, "r", encoding="utf8")
    lines = f.readlines()
    cnt = 0  # 第cnt个主题
    for line in lines:
        cnt += 1
        words = line.strip().split(' / ')
        for w in words:
            dic[cnt].append(w.strip())
    return dic


def split_into_sentences(text):  # 获得分句
    # 简单的文本预处理过程
    content = text
    content = content.lower()  # 转小写
    content = re.sub("[0-9]", "", content)  # 去掉数字
    content = content.replace("...", ".").replace("......", ".").replace("！", "!").replace("。", ".") \
        .replace("？", "?").replace("，", ",").replace(":", ",").replace("……", ".") \
        .replace(";", ",").replace(",", " ,").replace("!", ".").replace("?", ".").replace(".", " .")  # 调整符号
    content = content.replace("\n", "").replace("\t", "")  # 删去换行符和间隔符
    content = content.split(".")  # 依据.简单分句
    return content


def get_all_reviews():
    f = open(inputpath, "r", encoding="utf8")
    fo = open(resultpath, "a", encoding="utf8")
    lines = f.readlines()
    classification_model = SentimentOfSentence()
    min_max_scaler = preprocessing.MinMaxScaler()  # min_max 归一化
    pre_idx = 991943  # 上次运行到
    for i in tqdm(range(pre_idx+1, len(lines))):
        line = lines[i]
        parts = line.strip().split('\t')  # None表示信息缺失
        # print(parts)
        if len(parts) == 19:
            username, hours, products, product_id, date, text, early_access_user, found_funny, compensation, user_id, \
            genres, release_date, tags, discount_price, specs, price, early_access_game, sentiment, metascore = parts
        else:
            continue
        topic_sentiment_info = np.zeros((len(topic_dic.keys())+1, 1))  # review的主题-情感得分
        text = text.strip()
        sentence_list = split_into_sentences(text)  # 转化成分句
        for st in sentence_list:
            # print(st)  # st的长度不能超过maxlen
            words = st.split()
            topic_list = []
            for w in words:
                for k in topic_dic:  # 判断属于哪个主题
                    if w in topic_dic[k]:
                        if k not in topic_list:  # 不需重复加入
                            topic_list.append(k)
            topic_list = list(set(topic_list))
            if len(topic_list) == 0:
                continue

            # 获得情感极性
            classification_model.predict(st)
            polarity = classification_model.result  # 情感极性，0表示negative，1表示positive，2表示neutral

            if polarity == 0:  # 分数转换
                polarity = 0.2
            elif polarity == 2:
                polarity = 0.6
            elif polarity == 1:
                polarity = 1

            for topic in topic_list:
                topic_sentiment_info[topic] += polarity / len(topic_list)

        topic_sentiment_info = min_max_scaler.fit_transform(topic_sentiment_info).T  # 情感得分归一化
        line_out = username + "\t" + user_id + "\t" + product_id + "\t" + text \
                   + "\t" + str(list(topic_sentiment_info[0])) + "\n"
        fo.write(line_out)  # 输入到结果文件


def main():
    get_all_reviews()


if __name__ == "__main__":
    topic_dic = get_topics()
    main()
