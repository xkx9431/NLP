import re
from collections import Counter
from functools import reduce
from operator import mul,add
# import  matplotlib.pyplot  as plt
# from matplotlib.pyplot import yscale, xscale, title, plot,show

def tokenize(string):
    return ''.join(re.findall('[\w|\d]+', string))

all_content0=open('std_zh_wiki_00',encoding= 'utf-8').read()
# all_content1=open('std_zh_wiki_01',encoding= 'utf-8').read()
# all_content2=open('std_zh_wiki_02',encoding= 'utf-8').read()
all_char=tokenize(all_content0)

def n_gram_bag(content,gram_lenth=1):
    all_content_lenth = len(all_char)
    n_gram_bag = Counter(all_char[i:i + gram_lenth] for i in range(all_content_lenth - gram_lenth))
    return n_gram_bag


n1_char_counts = n_gram_bag(all_char)
n2_char_counts = n_gram_bag(all_char,2)


# M = all_char_counts.most_common()[0][1]
# yscale('log'); xscale('log'); title('Frequency of n-th most frequent word and 1/n line.')
# plot([c for (w, c) in all_char_counts.most_common()])
# plot([M/i for i in range(1, len(all_char_counts)+1)])
# show()

# print(type(all_char_counts))

def get_probability_from_counts(count1,count2):
    all_occurences1 = sum(count1.values())  # N for unigram bag
    all_occurences2 = sum(count2.values())  # N for 2-gram bag

    m_v1 = len(count1) # length for  the vocabulary uni
    m_v2 = len(count2) # length for  the vocabulary 2-gram
    def get_prob(item):
            if len(item)==1:
                return (1 + count1[item]) / (all_occurences1 + m_v1)
            else:
                prev=item[0]
                return (1 + count2[item]) / (count1[prev] + m_v2)
    return get_prob

get_prob = get_probability_from_counts(n1_char_counts,n2_char_counts)


def get_prob_sentence(sentence,n_gram=1):  #n_gram= set(1,2) only
    if n_gram==1:return reduce(mul,[get_prob(c)for c in sentence])
    else:
        probablities = []
        for i, c in enumerate(sentence):
            prev = '<s>' if i == 0 else sentence[i - 1]
            probablities.append(get_prob(prev+c))
        return reduce(mul,probablities)


def test():
    # print(get_char_prob('<s>')) # result=0
    # test pairs of sentence
    pair1 = """前天晚上吃晚饭的时候
    前天晚上吃早饭的时候""".split('\n')

    pair2 = """正是一个好看的小猫
    真是一个好看的小猫""".split('\n')

    pair3 = """我无言以对，简直
    我简直无言以对""".split('\n')
    pairs = [pair1, pair2, pair3]

    for pair in pairs:
        print('* *' * 8)
        for sentence in pair:
            print('\t{}with probility is {}'.format(sentence, get_prob_sentence(tokenize(sentence),2)))
test()