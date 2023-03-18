# 暂时用不到，通过get_topics生成的dict即可统计词频
# 通过分词、统计词频，提取主题
import string
from tqdm import tqdm
source_path = "data/reviews_generate_topics.txt"
save_path = "data/topics_from_word.txt"


def clean(text):
    stop_free = " ".join([i for i in text.split() if i not in stopword])
    punc_free = " ".join([ch for ch in stop_free.split() if ch not in exclude])  # 删掉标点
    return punc_free


def main():
    dictkeys = []  # 暂存关键词列表
    with open(source_path, encoding='utf8', mode='r') as f:
        lines = f.readlines()
        n = len(lines)
        for i in tqdm(range(n)):
            doc = lines[i]
            if not doc:
                break
            doc = str(doc).strip()
            word_clean = clean(doc).split()  # 去除停用词和标点后依据空格分词
            for word in word_clean:
                if word in dictkeys:
                    word_dic[word] += 1
                else:
                    word_dic[word] = 1
                    dictkeys.append(word)


def save_as_txt(dic):  # 词典存储为txt格式
    f = open(save_path, mode="w", encoding="utf8")
    wordlist = list(dic.items())
    print(wordlist)
    wordlist = sorted(wordlist, key=lambda t: t[1], reverse=True)
    print(wordlist)
    n = len(wordlist)
    for i in tqdm(range(n)):
        line = str(wordlist[i][0]) + "\t" + str(wordlist[i][1]) + "\n"
        f.write(line)


if __name__ == "__main__":
    # get stopword
    stopword = set()
    fr = open('stopwords.txt', mode='r', encoding='utf-8')
    for word in fr:
        stopword.add(word.strip())

    # get punctuation
    exclude = set(string.punctuation)  # 标点符号

    # word_dic = {'word': 1, 'dic': 2, 'test': 0}
    word_dic = {}
    main()
    save_as_txt(word_dic)