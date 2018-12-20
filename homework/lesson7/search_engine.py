import os
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from functools import reduce
from operator import and_
import re
from scipy.spatial.distance import cosine

vectorizer=TfidfVectorizer()
file_path='C:\\Users\\xkx\\Desktop\\NLP\\lesson7\\data'

def cut(string):return ' '.join(jieba.cut(string))
corpus=[
    cut(open(os.path.join(file_path,f),encoding='utf-8').read()) for f in os.listdir(file_path)[:1000]#简化计算
]
tfidf=vectorizer.fit_transform(corpus)
transposed_tfidf = tfidf.transpose()

transposed_tfidf_array=transposed_tfidf.toarray()#变成矩阵

def get_word_id(word):
    """
     after vectorizer.fit_transform(corpus)
    :return: word 在当前词典的索引号
    """
    return vectorizer.vocabulary_.get(word,None)
def get_candidates_ids(input_string):
    return [get_word_id(c) for c in cut(input_string).split()]
def get_candidates_pat(input_string):
    return '({})'.format('|'.join(cut(input_string).split()))

def search_enginer(query):
    """
    transposed_tfidf_array：row(word)*col(Doc)
    candidates: 找到每个候选词所在的文档编号
    """
    #
    candidats_ids=get_candidates_ids(query)
    v1=vectorizer.transform([cut(query)]).toarray()[0]
    candidates=[set(np.where(transposed_tfidf_array[_id])[0]) for _id in candidats_ids]
    merged_candidates=reduce(and_,candidates)
    pat =re.compile(get_candidates_pat(query))
    vector_with_id=[(tfidf[i],i) for i in merged_candidates ]
    #根据余弦相似度，对检索结果排序
    sorted_vector_with_ids=sorted(
        vector_with_id,key=lambda x:cosine(x[0].toarray(),v1))
    sorted_ids=[i for v,i in sorted_vector_with_ids]

    for c in sorted_ids:
        output=pat.sub(repl='***\g<1>**',string=corpus[c])
        yield ''.join(output.split())

def test():
    # print(len(os.listdir(file_path)))
    # print(vectorizer.fit_transform(corpus))
    # print(transposed_tfidf_array.shape)
    # print(np.where(transposed_tfidf_array[6]))
    for c in search_enginer('航空 飞机'):
        print(c)
test()


