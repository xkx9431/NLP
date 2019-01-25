# 采用决策树来判定新闻是不是来源于‘新华社’
# xkx-20190125
######
import pandas as pd
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier


#读取数据
content = pd.read_csv('sqlResult_1558435.csv', encoding='gb18030')
#观测数据
print(content.head())
print(content.info())#直观看到各特征属性中 数据空缺值数量

#数据清洗
content_source=content[['content','source']]
content_source.dropna(axis=0, how='any', inplace=True)

#查看 lable的分布情况
print(content_source['source'].value_counts())

#将 标签y 数值化,新华社为类1 其余为类0
content_source['source']=content_source['source'].apply(lambda source: 1 if source=='新华社'else 0 )
#print(content_source['source'].value_counts())
# 1    78661
# 0     8391
# Name: source, dtype: int64

#将输入文本转化为矩阵
def cut(s):return jieba.lcut(s)
def to_corpus(s):return ' '.join(cut(s)) if isinstance(s,str) else 0
content_source['content'] = content_source['content'].apply(to_corpus)
content_source = content_source[~content_source['content'].isin([0])]
vectorizer = TfidfVectorizer(max_features=30000)

X = vectorizer.fit_transform(content_source['content'])
Y = content_source['source']

## 随机抽取 33% 的数据作为测试集，其余为训练集
train_features, test_features, train_labels, test_labels = train_test_split(X, Y, test_size=0.33, random_state=0)
# 创建 CART 分类树
clf = DecisionTreeClassifier(criterion='gini')
# 拟合构造 CART 分类树
clf = clf.fit(train_features, train_labels)
# 用 CART 分类树做预测
test_predict = clf.predict(test_features)
# 预测结果与测试集结果作比对
score = accuracy_score(test_labels, test_predict)
print("CART 分类树准确率 %.4lf" % score)

# CART 分类树准确率 0.9889(会有差异)


