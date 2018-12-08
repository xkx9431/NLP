import jieba
import os
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence


#
# # #print(os.listdir())
# wiki=['wiki_simp_zh.txt']
#
# def write_jieb_f(openfile,output_file):
#     words=[]
#     for line in open(openfile,encoding='utf-8'):
#         w=list(jieba.cut(line))
#         words+= w +['\n']
#     output_file.writelines(' '.join(words))
#
# with open('jieba_wiki.txt','w',encoding='utf-8') as output_f:
#     for f in wiki:
#         write_jieb_f(f,output_f)

wiki_model = Word2Vec(LineSentence('wiki_zh_simp_seg.txt'),
                      size=200,window=5,min_count=5,workers=4)
wiki_model.save('wiki_w2v_model')
print(wiki_model.most_similar('北京'))