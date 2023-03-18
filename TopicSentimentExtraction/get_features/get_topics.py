# 本地构建词典，内存不够
import string
import time
import gensim
from gensim import corpora
import joblib
from tqdm import tqdm

source_path = "data/reviews_generate_topics.txt"


def clean(text):
    stop_free = " ".join([i for i in text.split(' ') if i not in stopword])
    punc_free = " ".join([ch for ch in stop_free.split(' ') if ch not in exclude])  # 删掉标点
    return punc_free


def main():
    LDA = gensim.models.ldamodel.LdaModel
    doclist = []  # 文档集合
    with open(source_path, encoding="utf8", mode="r") as f:
        # while True:
            # line = f.readline()
        lines = f.readlines()
        n = len(lines)
        print("读取评论：")
        for i in tqdm(range(n)):
            line = lines[i]
            if not line:
                break
            line = str(line).strip()
            doclist.append(line)

    print("\n开始数据清洗……")
    doc_clean = []
    for i in tqdm(range(len(doclist))):
        doc = doclist[i]
        doc_clean.append(clean(doc).split())

    # print("\n生成词典……")
    # dictionary = corpora.Dictionary(doc_clean)
    # dictionary.save('model/reviews_lda_v2.dict')
    # print("生成词典结束。")

    # 已生成词典，直接加载
    dictionary = corpora.Dictionary.load('model/reviews_lda_v2.dict')
    no_below = n * 0.001  # 过滤出现次数少于0.1%的
    no_above = 0.2  # 过滤出现频率高于20%的
    dictionary.filter_extremes(no_below=no_below, no_above=no_above, keep_n=10000)  # 过滤低频词和高频词，保留1w单词（实际上只有1000+单词）
    dictionary.save('model/reviews_dict_after_filtering_v2.dict')

    print("\n转换成doc_term_matrix：")
    doc_term_matrix = []
    for i in tqdm(range(len(doc_clean))):
        doc = doc_clean[i]
        doc_term_matrix.append(dictionary.doc2bow(doc))

    print("开始训练LDA模型……")
    num_topics = 10  # 主题数
    ldamodel = LDA(doc_term_matrix, num_topics=num_topics, id2word=dictionary, passes=50)  # num_topics 主题数
    print("LDA模型训练结束。")

    print("存储LDA模型……")
    savename = "model/lda_" + "topicnum=" + str(num_topics) + str(time.strftime('_%m_%d_%H_%M_%S', time.localtime(time.time()))) + ".pkl"
    joblib.dump(ldamodel, savename)
    print("LDA模型存储结束。")
    ldamodel.print_topics(num_topics=10, num_words=3)


if __name__ == "__main__":
    # get stopword
    stopword = set()
    fr = open('stopwords.txt', mode='r', encoding='utf-8')
    for word in fr:
        stopword.add(word.strip())
    # get punctuation
    exclude = set(string.punctuation)  # 标点符号
    main()
