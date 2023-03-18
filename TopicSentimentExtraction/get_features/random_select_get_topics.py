# 随机抽取50w条评论构建LDA主题模型（原始数据太大，构建主题模型所需时间太长）
# dict还是用整体review构建的
import math
import random
import string
import time
import gensim
from gensim import corpora
import joblib
from tqdm import tqdm
import matplotlib.pyplot as plt
source_path = "data/reviews_generate_topics.txt"
save_path = "data/random_select_reviews_50w.txt"
reviews_num = 500000


def graph_draw(topic, perplexity):  # 做主题数与困惑度的折线图
    x = topic
    y = perplexity
    plt.plot(x, y, color="red", linewidth=2)
    plt.xlabel("Number of Topic")
    plt.ylabel("Perplexity")
    plt.show()


def perplexity(ldamodel, testset, dictionary, size_dictionary, num_topics):  # 计算困惑度
    print('the info of this ldamodel: \n')
    print('num of topics: %s' % num_topics)
    prep = 0.0
    prob_doc_sum = 0.0
    topic_word_list = []
    for topic_id in range(num_topics):
        topic_word = ldamodel.show_topic(topic_id, size_dictionary)
        dic = {}
        for word, probability in topic_word:
            dic[word] = probability
        topic_word_list.append(dic)
    doc_topics_ist = []
    for doc in testset:
        doc_topics_ist.append(ldamodel.get_document_topics(doc, minimum_probability=0))
    testset_word_num = 0
    for i in range(len(testset)):
        prob_doc = 0.0  # the probablity of the doc
        doc = testset[i]
        doc_word_num = 0
        for word_id, num in dict(doc).items():
            prob_word = 0.0
            doc_word_num += num
            word = dictionary[word_id]
            for topic_id in range(num_topics):
                # cal p(w) : p(w) = sumz(p(z)*p(w|z))
                prob_topic = doc_topics_ist[i][topic_id][1]
                prob_topic_word = topic_word_list[topic_id][word]
                prob_word += prob_topic * prob_topic_word
            prob_doc += math.log(prob_word)  # p(d) = sum(log(p(w)))
        prob_doc_sum += prob_doc
        testset_word_num += doc_word_num
    prep = math.exp(-prob_doc_sum / testset_word_num)  # perplexity = exp(-sum(p(d)/sum(Nd))
    print("模型困惑度的值为 : %s" % prep)
    return prep


def clean(text):
    stop_free = " ".join([i for i in text.split(' ') if i not in stopword])
    punc_free = " ".join([ch for ch in stop_free.split(' ') if ch not in exclude])  # 删掉标点
    return punc_free


def random_select(num, path):
    f = open(path, encoding='utf8', mode='r')
    lines = list(f.readlines())
    print("原始评论共有{0}条。".format(len(lines)))
    random.seed(77)
    sample = random.sample(lines, num)
    sample = [s.strip() for s in sample]
    print("随机抽取后的评论共有{0}条。".format(len(sample)))
    # print(sample)

    print("开始存储随机抽取的评论……")
    fo = open(save_path, encoding="utf8", mode="w")
    for s in sample:
        fo.write(s+"\n")
    fo.close()
    print("存储结束。")
    return sample


def main():
    data = random_select(reviews_num, source_path)
    print("\n开始数据清洗……")
    doc_clean = []
    for i in tqdm(range(len(data))):
        doc = data[i]
        doc_clean.append(clean(doc).split())
    dictionary = corpora.Dictionary.load('model/reviews_dict_after_filtering_v2.dict')
    corpus = [dictionary.doc2bow(text) for text in
              doc_clean]  # corpus里面的存储格式（0, 1), (1, 1), (2, 1), (3, 1), (4, 1), (5, 1), (6, 1)
    corpora.MmCorpus.serialize('model/corpus.mm', corpus)

    print("\n转换成doc_term_matrix：")
    doc_term_matrix = []
    for i in tqdm(range(len(doc_clean))):
        doc = doc_clean[i]
        doc_term_matrix.append(dictionary.doc2bow(doc))

    LDA = gensim.models.ldamodel.LdaModel
    num_topics_list = [10, 15, 20, 25, 30, 35, 40, 45, 50]  # 候选主题数
    prep_list = []  # 存储困惑度
    for num_topics in num_topics_list:
        print("num_topics={0}".format(str(num_topics)))
        print("开始训练LDA模型……")
        ldamodel = LDA(doc_term_matrix, num_topics=num_topics, id2word=dictionary, passes=50)  # num_topics 主题数
        print("LDA模型训练结束。")

        print("存储LDA模型……")
        savename = "model/lda_" + "topicnum=" + str(num_topics) + str(time.strftime('_%m_%d_%H_%M_%S', time.localtime(time.time()))) + ".pkl"
        joblib.dump(ldamodel, savename)
        print("LDA模型存储结束。")

        print("展示主题-词分布：")
        print(ldamodel.print_topics(num_topics=10, num_words=10))

        corpus = corpora.MmCorpus('model/corpus.mm')  # 加载语料库迭代器
        testset = []
        for c in range(int(corpus.num_docs)):
            testset.append(corpus[c])
        prep = perplexity(ldamodel, testset, dictionary, len(dictionary.keys()), num_topics)
        prep_list.append(prep)
        # print("困惑度是{0}".format(str(prep)))
    graph_draw(num_topics_list, prep_list)  # 绘制困惑度曲线


if __name__ == "__main__":
    # get stopword
    stopword = set()
    fr = open('stopwords.txt', mode='r', encoding='utf-8')
    for word in fr:
        stopword.add(word.strip())

    # get punctuation
    exclude = set(string.punctuation)
    main()