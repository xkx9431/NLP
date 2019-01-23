import pandas as pd
import numpy as np
import jieba
import random
from sklearn.feature_extraction.text import  TfidfVectorizer
from sklearn.cluster import KMeans
from collections import  defaultdict

#读取数据
movie_content ='movie_comments.csv'
content = pd.read_csv(movie_content)
all_comments = content['comment'].tolist()
# 分词
def cut(string):return jieba.lcut(string)
all_comments = [' '.join(cut(s)) for s in all_comments if isinstance(s,str)]

samples = random.sample(all_comments,1000)
vectorizer = TfidfVectorizer(max_features=30000)

X = vectorizer.fit_transform(samples)
id_word = {i: w for w, i in vectorizer.vocabulary_.items()}

#Kmeans 聚类拟合
kmeans = KMeans(n_clusters=30,random_state=0).fit(X)

all_lables = kmeans.labels_.tolist()
lables_with_comment = defaultdict(list)
for i, label in enumerate(all_lables):
    lables_with_comment[label].append(samples[i])




def test():
    print(samples[1])
    print(np.where(X[1].toarray()[0]))
    print(kmeans.labels_)
    for label in lables_with_comment:
        print('lable : {}'.format(label))
        for comment in lables_with_comment[label]:
            print('\t: {}'.format(comment))
test()
