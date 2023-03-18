# 0911 预处理评论数据集，以便进行主题生成
# 0912 调整预处理方式，重新处理评论
import re
from tqdm import tqdm
import pandas as pd
source_path = "../data/csv/steam_new.csv"
target_path = "data/reviews_generate_topics.txt"  # 0912 version


def get_reviews(data):
    f = open(target_path, encoding="utf8", mode="w")
    n = len(data)
    print("begin generating reviews……")
    linecnt = 0
    for i in tqdm(range(n)):
        content = str(data.loc[i, "text"]).strip()
        # 简单的文本预处理过程
        content = content.lower()  # 转小写
        content = re.sub("[0-9]", "", content)  # 去掉数字
        content = content.replace("...", ".").replace("......", ".").replace("！", "!").replace("。", ".")\
            .replace("？", "?").replace("，", ",").replace(":", ",").replace("……", ".")\
            .replace(";", ",").replace(",", " ,").replace("!", ".").replace("?", ".").replace(".", " .")  # 调整符号
        content = content.replace("\n", "").replace("\t", "")  # 删去换行符和间隔符
        content = content.split(".")  # 依据.简单分句
        # print(content)
        for j in range(len(content)):
            ct = content[j].strip()
            if len(ct) == 0:
                continue
            # if ct[-1] != "," and ct[-1] != "!":
            #    ct = ct.strip() + "."  # 恢复.
            line = ct + "\n"
            # print(line)
            f.write(line)
            linecnt += 1
    print("linecnt: ", linecnt)
    print("finished.")
    f.close()


if __name__ == "__main__":
    data = pd.read_csv(source_path)
    get_reviews(data)