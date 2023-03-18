# 绘制主题-困惑度曲线
# 从训练好的LDA模型中提取主题信息
import os
import joblib
from gensim import corpora
import matplotlib.pyplot as plt
from get_features.random_select_get_topics import perplexity, graph_draw
ldamodel_path = "model/ldamodels"


def get_dict():
    d = corpora.Dictionary.load('model/reviews_dict_after_filtering_v2.dict')
    return d


def save_topics(topics):
    for idx, topic in topics:
        words = [w.strip().split("*") for w in topic.split("+")]
        line = "num_words=" + str(len(words)) + "\t"
        for rate, w in words:
            line += str(w)[1:-1] + " " + str(rate) + "\t"
        line = line.strip() + "\n"
        print(line)
        f.write(line)
    f.write("\n")


def main():
    dictionary = get_dict()
    corpus = corpora.MmCorpus('model/corpus.mm')  # 加载语料库迭代器
    testset = []
    for c in range(int(corpus.num_docs / 10)):
        testset.append(corpus[c*10])

    ldamodel_list = []
    num_topics_list = []
    prep_list = []  # 困惑度列表
    filenames = os.listdir(ldamodel_path)
    for file in filenames:
        path = ldamodel_path + "/" + file
        model = joblib.load(path)
        ldamodel_list.append(model)
        num_topics_list.append(int(file.split("=")[1][:2]))
        prep = perplexity(model, testset, dictionary, len(dictionary.keys()), int(file.split("=")[1][:2]))
        prep_list.append(prep)
    graph_draw(num_topics_list, prep_list)  # 绘制困惑度曲线

    for idx in range(len(ldamodel_list)):
        model = ldamodel_list[idx]
        num_topics = num_topics_list[idx]
        topics = model.print_topics(num_topics=num_topics)
        # print(len(topics))
        f.write("num_topics=" + str(num_topics) + "\n")
        save_topics(topics)


if __name__ == "__main__":
    f = open("model/ldamodels_info.txt", encoding="utf8", mode="w")  # 存储lda模型的主题提取信息
    main()