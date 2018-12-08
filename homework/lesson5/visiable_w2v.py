####Visualizing Word Vectors with t-SNE

import numpy as np
import re
# import nltk
from gensim.models import Word2Vec
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

model =  Word2Vec.load('wiki_w2v_model')
# print(model.wv.similarity('奥运会','金牌'))

#查看词向量维度
# y1=model.vector_size
# print(y1)

#模型可视化
def vis_word_vec(model,n):
    count=1
    tokens=[]
    labels=[]
    for word in model.wv.vocab:
        tokens.append(model.wv[word])
        labels.append(word)
        count+=1
        if count>=n:
            break

    tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500, random_state=23)
    new_values = tsne_model.fit_transform(tokens)

    x = []
    y = []
    for value in new_values:
        x.append(value[0])
        y.append(value[1])

    plt.figure(figsize=(20, 20))
    for i in range(len(x)):
        plt.scatter(x[i], y[i])
        plt.annotate(labels[i],
                     xy=(x[i], y[i]),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
    plt.show()

#
vis_word_vec(model,100)
