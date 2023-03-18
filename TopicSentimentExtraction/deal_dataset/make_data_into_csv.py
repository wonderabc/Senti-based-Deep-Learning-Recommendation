# 将json数据转换为csv格式
import json
import pandas as pd
import gzip

def parse(path):
  g = gzip.open(path, 'rb')
  for l in g:
    # print(l)
    yield json.loads(l)

def getDF(path):
  i = 0
  df = {}
  for d in parse(path):
    df[i] = d
    i += 1
  return pd.DataFrame.from_dict(df, orient='index')


if __name__ == "__main__":
  # review_df = getDF('./data/Software_5.json.gz')
  # meta_df = getDF('./data/meta_Software.json.gz')
  review_df = getDF('./data/Magazine_Subscriptions_5.json.gz')
  meta_df = getDF('./data/meta_Magazine_Subscriptions.json.gz')
  print("评论共{}条，meta数据共{}条。".format(len(review_df), len(meta_df)))
  # review_df.to_csv('./data/csv/softwares_reviews_5core.csv', index=False)
  # meta_df.to_csv('./data/csv/softwares_meta.csv', index=False)
  review_df.to_csv('./data/csv/magazines_reviews_5core.csv', index=False)
  meta_df.to_csv('./data/csv/magazines_meta.csv', index=False)
