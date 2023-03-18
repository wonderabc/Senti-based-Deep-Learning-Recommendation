# 根据构建LDA模型生成的dict读取词频信息
from gensim import corpora
from tqdm import tqdm

save_path = "data/topics_from_words_no_filtering.txt"


if __name__ == "__main__":
    f = open(save_path, encoding="utf8", mode="w")
    # dictionary = corpora.Dictionary()
    dictionary = corpora.Dictionary.load("model/reviews_lda.dict")  # 加载dictionary
    # id2token = dictionary.id2token
    # print(id2token)
    # print(dictionary[1])
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
    f.close()

