# 生成训练数据
import json
import demjson
from tqdm import tqdm

data_path = "../GBDT_LR/data/1128_preprocessed_generated_include_topic_feature.txt"
amazon_data_path = "../data/amazon/records_include_gameanduser_info_0109.txt"
software_data_path = "../data/software/software_all_record_0504.txt"
magazine_data_path = "../data/magazine/magazine_all_record_0504.txt"

save_path = "./data/u.data"  # save_path of steam
amazon_save_path = "./data/amazon_u.data"  # save_path of amazon
software_save_path = "./data/software_u.data"  # save_path of amazon
magazine_save_path = "./data/magazine_u.data"

def main():
  f = open(data_path, "r", encoding="utf8")
  f_save = open(save_path, "w", encoding="utf8")
  lines = f.readlines()
  for i in tqdm(range(len(lines))):
    line = lines[i].strip()
    line = demjson.encode(line)
    tempinfo = eval(json.loads(line))
    userid = tempinfo['user_id']
    productid = tempinfo['product_id']
    label = tempinfo['Label']
    tmp_line = str(userid) + "\t" + str(productid) + "\t" + str(label) + "\n"
    f_save.write(tmp_line)


def main_amazon():
  f = open(amazon_data_path, "r", encoding="utf8")
  f_save = open(amazon_save_path, "w", encoding="utf8")
  lines = f.readlines()
  for line in lines:
    parts = line.strip().split("\t")
    if len(parts) != 17:
      continue
    reviewerID, asin, reviewerName, helpful, reviewText, overall, summary, unixReviewTime, \
    reviewTime, product_description, product_price, product_related, product_salesRank, \
    product_categories, product_title, product_brand, user_categories = parts
    reviewerID = reviewerID.strip()
    asin = asin.strip()
    score = float(overall.strip())
    tmp_line = str(reviewerID) + "\t" + str(asin) + "\t" + str(score) + "\n"
    f_save.write(tmp_line)


def main_general(path1, path2):
  f = open(path1, "r", encoding="utf8")
  f_save = open(path2, "w", encoding="utf8")
  lines = f.readlines()
  for line in lines:
    parts = line.strip().split("\t")
    if len(parts) == 9:
      overall, reviewerID, asin, reviewerName, reviewText, vote, productbrand, productrank, productprice = parts
    else:
      continue
    asin = asin.strip()
    reviewerID = reviewerID.strip()
    score = float(overall.strip())
    tmp_line = str(reviewerID) + "\t" + str(asin) + "\t" + str(score) + "\n"
    f_save.write(tmp_line)


if __name__ == "__main__":
  # main()
  # main_amazon()
  main_general(magazine_data_path, magazine_save_path)