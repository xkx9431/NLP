import re
from collections import Counter
def tokenize(string):
    return ''.join(re.findall('[\w|\d]+', string))

all_content=open('80k_articles.txt',encoding= 'utf-8').read()
all_char=tokenize(all_content)

gram_lenth=2
all_content_lenth=len(all_char)
# print(len(all_char))
all_char_counts=Counter(all_char[i:i+gram_lenth]for i in range(all_content_lenth-gram_lenth))
# print(type(all_char_counts))

def get_probability_from_counts(count):
    all_occurences = sum(count.values()) # N
    threshold = 5 # r小于该阈值，需要修正
    min_nc = [0] * (threshold+1)
    #计算频数r=1,2..threshold 的词数min_nc[r]
    for i in count:
        if count[i] <= threshold:
            min_nc[count[i]] += 1

    def get_prob(item):
        if item not in count:
            return count[1]/all_occurences # p0 for unseen
        if count[item]>=threshold:
            return count[item] / all_occurences # r is big enough
        else:  # need Good
            r = count[item]
            dr = (1+r)*S(min_nc[r+1])/S(min_nc[r])
            if dr<r:
                return dr / all_occurences
            else:
                return r / all_occurences

    def S(item): return 1+item      # empirical Bayes method
    return get_prob

get_char_prob = get_probability_from_counts(all_char_counts)

print(get_char_prob('你好'))