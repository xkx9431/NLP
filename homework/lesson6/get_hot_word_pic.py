import jieba
import pandas as pd
from collections import Counter
import math
import re
from  tqdm import tqdm
import  wordcloud
import  numpy as np
from PIL import  Image
from matplotlib.pyplot import show
import matplotlib.pyplot as plt


content = pd.read_csv('sqlResult_1558435.csv', encoding='gb18030')
content = content.fillna('')
all_news_content=content.iloc[0:100]['content']
all_occurences=[]

def cut(string): return list(jieba.cut(string))

content_of_xiaomi = cut(content.iloc[0]['content'])

for c in tqdm(all_news_content):
    all_occurences.append(set(cut(c)))

def inverse_document_frequency(word):
    eps=1e-6
    return math.log10(len(all_occurences)/(sum(1 for w in all_occurences if word in w)+1))
def term_frequency(word,cut_word_counter):
    return cut_word_counter[word]/sum(cut_word_counter.values())


def tfidf(word, cut_words_counter):
    w_tf = term_frequency(word, cut_words_counter)
    idf = inverse_document_frequency(word)
    return w_tf * idf
def get_important_word(cut_words):
    importance={w:tfidf(w,Counter(cut_words)) for w in set(cut_words)}
    return sorted(importance.items(),key=lambda x:x[1],reverse=True)
#
xiaomi_important= get_important_word(content_of_xiaomi)
xiaomi_mask = np.array(Image.open('hot_pic.png'))
word_cloud_with_mask = wordcloud.WordCloud(font_path='E:\\NLP\\SimHei.ttf',mask=xiaomi_mask)

def plot_word_cloud_by_importance(importance):
    plt.imshow(word_cloud_with_mask.generate_from_frequencies({w: fre for w, fre in importance}))
plot_word_cloud_by_importance(xiaomi_important)
word_cloud_with_mask.generate_from_frequencies({w: fre for w, fre in xiaomi_important}).to_file('xiaomi_hot.png')

# def test():
#     # print(tfidf('小米',Counter(content_of_xiaomi)))
#     # print(tfidf('的',Counter(content_of_xiaomi)))
#     # print(get_important_word(content_of_xiaomi))
#     # #print(Counter(content_of_xiaomi).most_common()[:10])
#
# test()

