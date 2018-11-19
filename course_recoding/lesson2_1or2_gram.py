# given pairs of sentence,and determine which are normal
# import os
# print(os.listdir())
## read a word lib '80k_articles.txt'
import re
from collections import Counter
from functools import reduce
from operator import add,mul
all_content=open('80k_articles.txt',encoding= 'utf-8').read()

# remove useless strings in content
def tokensize(string):
    return ''.join(re.findall('[\w|\d]+',string))

all_chr=tokensize(all_content)
all_chr_counts=Counter(all_chr)
####### Uigram########
# calc the probability of every char and the base is the content
def get_all_probability_form_base(base):
    all_counts=sum(base.values())
    def get_prob_char(item):  # efficient for calc sum 1 times
        return base[item]/all_counts
    return get_prob_char

prob_char_of = get_all_probability_form_base(all_chr_counts)

def get_prob_of_sen(string):
    return reduce(mul,[prob_char_of(c) for c in string])

#test pairs of sentence
pair1 = """前天晚上吃晚饭的时候
前天晚上吃早饭的时候""".split('\n')

pair2 = """正是一个好看的小猫
真是一个好看的小猫""".split('\n')

pair3 = """我无言以对，简直
我简直无言以对""".split('\n')
pairs=[pair1,pair2,pair3]

def get_result_prob_sen(func_version,pairs):
    for sen1,sen2 in pairs:
        print('* '*10)
        print('\t{}with probability {}'.format(sen1,func_version(tokensize(sen1))))
        print('\t{}with probability {}'.format(sen2,func_version(tokensize(sen2))))




####### use 2 gram #######
gram_lenth=2
two_gram_counts=Counter(all_chr[i:i+gram_lenth] for i in range(len(all_chr)-gram_lenth))
get_pair_prob=get_all_probability_form_base(two_gram_counts)

#
def get_2_gram_prob(word,prev):
    if get_pair_prob(word+prev)>0:
        return get_pair_prob(word+prev)/prob_char_of(prev)
    else:
        return prob_char_of(word)

def get_2gram_sen_prob(sen):
    prob=[]
    for i,c in enumerate(sen):
        prev='<s>' if i==0 else sen[i-1]
        prob.append(get_2_gram_prob(c,prev))
    return reduce(mul,prob)

# unigram vs 2-gram
get_result_prob_sen(get_prob_of_sen,pairs)
get_result_prob_sen(get_2gram_sen_prob,pairs)




# def test():
#     print(all_content[:200])
#     print(len(all_chr))
#     print(prob_char_of('我'))
#
# test()