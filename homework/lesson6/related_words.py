from gensim.models import Word2Vec
from collections import defaultdict
model=Word2Vec.load('wiki_with_news_model')

def get_related_words(initial_word,model):
    max_size=100
    seen=defaultdict(int)
    unseen=[initial_word]
    layer=1
    while unseen and len(seen)<max_size:
        if len(seen)%20==0:
            print('seen length:{}'.format(len(seen)))
        node =unseen.pop(0)
        new_expandng=[w for w,s in model.most_similar(node,topn=10)]
        unseen +=new_expandng
        seen[node]+=1  # 1 为重要性,具体score,可以根据当前搜索层数，相似度改变
    return seen

shuo=get_related_words('说',model)
with open('shuo.txt', 'w',encoding='utf-8') as f:
    for e in shuo:
        f.write(e+"\n")
    f.close()