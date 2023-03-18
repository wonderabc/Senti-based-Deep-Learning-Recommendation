import ast
from collections import defaultdict
from deal_dataset.get_topics_bydict import getmetas, getreviews
import numpy as np

software_meta_columns = ['category', 'tech1', 'description', 'fit', 'title', 'also_buy', 'tech2', 'brand', 'feature', 'rank', 'also_view', 'main_cat', 'similar_item', 'date', 'price', 'asin', 'imageURL', 'imageURLHighRes', 'details']
software_review_columns = ['overall', 'verified', 'reviewTime', 'reviewerID', 'asin', 'style', 'reviewerName', 'reviewText', 'summary', 'unixReviewTime', 'vote', 'image']
magazine_meta_columns = ['category', 'tech1', 'description', 'fit', 'title', 'also_buy', 'tech2', 'brand', 'feature', 'rank', 'also_view', 'details', 'main_cat', 'similar_item', 'date', 'price', 'asin', 'imageURL', 'imageURLHighRes']
magazine_review_columns = ['overall', 'verified', 'reviewTime', 'reviewerID', 'asin', 'reviewerName', 'reviewText', 'summary', 'unixReviewTime', 'vote', 'style', 'image']
software_highfre_catelist = ['Software', "Children's", 'Education & Reference', 'Digital Software', 'Business & Office', 'Lifestyle & Hobbies', 'Utilities', 'Languages', '</span></span></span>', 'Home Publishing', 'Antivirus & Security', 'Design & Illustration', 'Education &amp; Reference', 'Accounting & Finance', 'Music', 'Games', 'Early Learning', 'Video', 'Operating Systems', 'Photography', 'Programming & Web Development', 'Business &amp; Office', 'Antivirus', 'Backup', 'Training', 'Document Management', 'Personal Finance', 'Internet Security Suites', 'Maps & Atlases', 'Lifestyle &amp; Hobbies', 'Office Suites', 'Business Accounting', 'PC Maintenance', 'Video Editing', 'Web Design', 'Clip Art', 'Networking & Servers', 'Instrument Instruction', 'CAD', 'Test Preparation', 'Dictionaries', 'Animation & 3D', 'Word Processing', 'Microsoft Windows', 'Communication', 'Money Management & Budgeting', 'Illustration', 'MP3 Editing & Effects', 'Linux & Unix', 'DVD Viewing & Burning', 'Home & Garden Design', 'Contact Management', 'Internet Utilities', 'Mac Operating Systems', 'Tax Preparation', 'Greeting Cards', 'Design &amp; Illustration', 'Accounting &amp; Finance', 'Database', 'Math', 'Sound Libraries', 'CD Burning & Labeling', 'Programming &amp; Web Development', 'Security', 'Business & Marketing Plans', 'Business Planning', 'Religion', 'Drivers & Driver Recovery', 'Genealogy', 'Maps &amp; Atlases', 'Art & Creativity', 'Typing', 'Cooking & Health', 'Science', 'Music Notation', 'Legal', 'Programming Languages', 'Voice Recognition', 'Project Management', 'Development Utilities', 'Reading & Language', 'Web Effects', 'Training & Tutorials', 'Visualization & Presentation', 'Web Page Editors', 'Compositing & Effects', 'Screen Savers', 'Check Printing', 'Fonts', 'Firewalls', 'Antivirus &amp; Security', 'Networking &amp; Servers', 'E-mail', 'File Conversion', 'Servers', 'Science & Nature', 'Internet Security', 'Cross Platform', 'Scrapbooking', 'Calendars']
magazine_highfre_catelist = ['Magazine Subscriptions', 'Professional & Educational Journals', 'Professional & Trade', 'Sports, Recreation & Outdoors', 'Sports & Leisure', 'United States', 'Professional &amp; Educational Journals', 'Professional &amp; Trade', 'Travel, City & Regional', 'Humanities & Social Sciences', 'Arts', 'Fashion & Style', 'Home & Garden', 'Entertainment & Pop Culture', 'History', 'Transportation', 'Arts, Music & Photography', 'Religion & Spirituality', 'Sports, Recreation &amp; Outdoors', 'Sports &amp; Leisure', 'South', 'Children', 'Cooking, Food & Wine', 'Music', 'Children & Teen', 'Humanities &amp; Social Sciences', 'Women', 'By Age', 'Travel, City &amp; Regional', 'Lifestyle & Cultures', 'West', 'Business & Investing', 'Engineering', 'Automotive & Motorcycles', 'Crafts & Hobbies', 'Literary, Sci-Fi & Mystery', 'Ages 9-12', 'Hunting & Firearms', 'Northeast', 'Technology', 'Automotive', 'Photography', 'Science & Technology', 'Health, Fitness & Wellness', 'Entertainment & Media', 'Science, History & Nature', 'Midwest', 'Design & Decoration', 'Architecture', 'Education', 'Recipes & Techniques', 'Health', 'Literary Magazines & Journals', 'Boating', 'Christianity', 'Medicine', 'Science &amp; Technology', 'Linguistics', 'Flying', 'Economics & Economic Theory', 'Antiques & Collectibles', 'Equestrian', 'News & Political Commentary', 'Computers & Internet', 'Pets & Animals', 'Training', 'Puzzles & Games', 'Biological & Natural Sciences', 'Decorative Arts', 'Movies', 'Crafts & Collectibles', 'Extreme Sports', "Women's Interest", 'Men', 'Religion &amp; Spirituality', 'International', 'Military Science', 'Ideas & Commentary', 'Children &amp; Teen', 'Science Fiction & Fantasy', 'Drawing, Design & Illustration', 'Arts, Music &amp; Photography', 'Home &amp; Garden', 'Crafts &amp; Hobbies', 'Hunting &amp; Firearms', 'Special Diet', 'Entertainment &amp; Pop Culture', 'Business &amp; Investing', 'Law', 'Water Sports', 'Celebrities & Gossip', 'Ages 4-8', 'Gardening & Horticulture', 'Finance', 'Biological &amp; Natural Sciences', 'Golf', 'Cooking, Food &amp; Wine', 'Parenting & Families', 'Woodworking', "Men's Interest"]

software_record_save_path = "./data/software_all_record_0504.txt"
magazine_record_save_path = "./data/magazine_all_record_0504.txt"
software_user_save_path = "./data/software_userfeature_0504.txt"
magazine_user_save_path = "./data/magazine_userfeature_0504.txt"
software_cate_save_path = "./data/software_products_cate_0506.txt"
magazine_cate_save_path = "./data/magazine_products_cate_0506.txt"
software_usercate_save_path = "./data/software_user_cate_0507.txt"
magazine_usercate_save_path = "./data/magazine_user_cate_0507.txt"


def main():
  metas = getmetas()
  reviews = getreviews()
  print(len(metas))
  print("software meta example: ")
  print(metas.loc[100])
  # print(metas["details"][:100])
  # for i in range(len(metas)):
    # details = str(metas.loc[i, "details"])
    # if len(details) > 0:
      # print(details)
  # metas["feature"] = metas["feature"].astype(str)
  # print(len(set(metas["feature"])))

  print("software review example: ")
  print(reviews.loc[100])
  print(reviews["vote"][:1000])


def get_categories():  # 获取类别信息
  metas = getmetas()
  category_dic = defaultdict(int)
  for i in range(len(metas)):
    category_info = ast.literal_eval(str(metas.loc[i, "category"]))
    for cate in category_info:
      category_dic[cate] += 1
    # brand = str(metas.loc[i, "brand"])
    # brand = brand.replace("&amp;", "&").replace("\n", "").replace("by ", " ").strip()
    # category_dic[brand] += 1
  # print(list(category_dic.keys()))
  print(len(list(category_dic.keys())))
  category_list = sorted(category_dic.items(), key=lambda t:t[1], reverse=True)
  cnt = 0
  highfre = []
  for cate in category_list:
    cnt += 1
    print(cnt, cate)
    highfre.append(cate[0])
    if cnt == 100:
      break
  print(highfre)


def get_user_features():  # 获得用户历史信息
  f = open(magazine_record_save_path, "r", encoding="utf8")
  lines = f.readlines()
  # print(len(lines))
  user_dic = defaultdict(lambda: {"rank": 0, "price":0.0, "cnt": 0})
  for line in lines:
    if len(line.strip().split("\t")) != 9:
      continue
    overall, reviewerID, asin, reviewerName, reviewText, vote, productbrand, productrank, productprice = line.strip().split("\t")
    reviewerID = reviewerID.strip()
    user_dic[reviewerID]["cnt"] += 1
    user_dic[reviewerID]["rank"] += int(productrank)
    user_dic[reviewerID]["price"] += float(productprice)

  f_save = open(magazine_user_save_path, "w", encoding="utf8")
  for userid in user_dic:
    average_rank = float(user_dic[userid]["rank"]) / float(user_dic[userid]["cnt"])
    average_price = float(user_dic[userid]["price"]) / float(user_dic[userid]["cnt"])
    tempstr = str(userid) + "\t" + str(average_rank) + "\t" + str(average_price) + "\n"
    f_save.write(tempstr)


def generate_trainset():
  metas = getmetas()
  reviews = getreviews()
  metadict = defaultdict(lambda: {})
  for i in range(len(metas)):
    asin = str(metas.loc[i, "asin"]).strip()  # key

    brand = metas.loc[i, "brand"]
    brand = brand.replace("&amp;", "&").replace("\n", "").replace("by ", " ").strip()
    rank = metas.loc[i, "rank"]
    try:
      rank = int(rank.replace(" in Software (", "").replace(",", ""))
    except:
      rank = 0
    price = metas.loc[i, "price"]
    try:
      price = float(price.replace("$", "").strip())
    except:
      price = 0.0
    tempinfo = {"brand": brand, "rank": rank, "price": price}
    metadict[asin] = tempinfo
  cnt = 0
  f = open(magazine_record_save_path, "w", encoding="utf8")
  for i in range(len(reviews)):
    overall = float(reviews.loc[i, "overall"])
    reviewerID = str(reviews.loc[i, "reviewerID"])
    asin = str(reviews.loc[i, "asin"])
    reviewerName = str(reviews.loc[i, "reviewerName"])
    reviewText = str(reviews.loc[i, "reviewText"])
    reviewText = reviewText.replace("\n", " ")
    try:
      vote = int(str(reviews.loc[i, "vote"]).strip())
    except:
      vote = 0
    if asin not in metadict:
      pass
    else:
      cnt += 1
      productbrand = metadict[asin]["brand"]
      productrank = metadict[asin]["rank"]
      productprice = metadict[asin]["price"]
      tempstr = str(overall) + "\t" + str(reviewerID) + "\t" + str(asin) + "\t" + str(reviewerName) + "\t" + str(reviewText) + \
                "\t" + str(vote) + "\t" + str(productbrand) + "\t" + str(productrank) + "\t" + str(productprice) + "\n"
      f.write(tempstr)

  print("记录共有{}条。".format(cnt))


def generate_catedict():  # 生成product类别词典，存储在文件中
  metas = getmetas()
  f = open(magazine_cate_save_path, "w", encoding="utf8")
  for i in range(len(metas)):
    asin = metas.loc[i, "asin"]
    categorylist = ast.literal_eval(str(metas.loc[i, "category"]).strip())
    tmpstr = str(asin).strip() + "\t"
    catedict = defaultdict(int)
    for cate in categorylist:
      if cate in magazine_highfre_catelist:
        catedict[cate] = 1
    for cate in magazine_highfre_catelist:
      tmpstr += str(catedict[cate]) + "\t"
    tmpstr = tmpstr.strip() + "\n"
    f.write(tmpstr)

def generate_user_catedict():  # 生成user类别词典，存储在文件中
  metas = getmetas()
  product_catedict = {}
  for i in range(len(metas)):
    asin = str(metas.loc[i, "asin"]).strip()
    categorylist = ast.literal_eval(str(metas.loc[i, "category"]).strip())
    catedict = defaultdict(int)
    for cate in categorylist:
      if cate in magazine_highfre_catelist:
        catedict[cate] = 1
    cateinfo = [catedict[cate] for cate in magazine_highfre_catelist]
    cateinfo = np.array(cateinfo).reshape((1, len(magazine_highfre_catelist)))
    # print(cateinfo)
    product_catedict[asin] = cateinfo

  reviews = getreviews()
  user_catedict = defaultdict(lambda: np.zeros((1, 100)))
  user_pnum_dict = defaultdict(int)  # user关联的产品数词典
  form_err = 0
  for i in range(len(reviews)):
    asin = str(reviews.loc[i, "asin"]).strip()
    userid = str(reviews.loc[i, "reviewerID"]).strip()
    if asin in product_catedict:
      user_catedict[userid] += product_catedict[asin]
      user_pnum_dict[userid] += 1
    else:
      form_err += 1

  print("缺失asin信息的记录有{}条。".format(form_err))
  for userid in user_catedict:
    user_catedict[userid] = user_catedict[userid] / float(user_pnum_dict[userid])

  # print(sorted(user_pnum_dict.items(), key=lambda t:t[1], reverse=True))
  # 存储user cate信息
  f_save = open(magazine_usercate_save_path, "w", encoding="utf8")
  for userid in user_catedict:
    catelist = list(user_catedict[userid][0])
    # print(catelist)
    tmpstr = userid + "\t"
    for cate in catelist:
      tmpstr += str(cate) + "\t"
    tmpstr = tmpstr.strip() + "\n"
    f_save.write(tmpstr)


if __name__ == "__main__":
  # main()
  # get_categories()
  # category为list形式，共6956个category --> not useful
  # brand 一对一，共4259个brand（清洗后） --> useful;
  # brand = brand.replace("&amp;", "&").replace("\n", "").replace("by ", " ").strip()

  # rank useful; rank = rank.replace(" in Software (", "").replace(",", "")
  # date not useful;
  # price useful; price = float(price.replace("$", "").strip())
  # asin useful; 商品号
  # details not useful

  # features of reviews
  # overall 1.0 - 5.0 useful; label
  # reviewerID / asin / reviewerName / reviewText  useful; basic information
  # vote 是否需要考虑？; num of voting for the review

  # generate_trainset()
  # get_user_features()
  # generate_catedict()
  generate_user_catedict()