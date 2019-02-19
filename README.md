# NLP course
nlp 课程  目录
## 课程1 An introduction to AI
#### 人工智能概述，发展历史，
#### 语法树相关知识
## 课程2 概率模型
#### 概率模型
#### 机器学习模型引论
#### 自动决策模型
#### 自动机理论
作业：下载维基语料库 ，尝试比较相似语句在不同语言模型概率大小
## 课程3 智能搜索
#### 深度搜索、广度搜索、启发式搜索
作业：实现北京自动地铁换乘系统（车站信息获取，不同换乘策略实现）
## 课程4 动态规划
#### 动态规划方法论
#### 动态规划实例——背包问题
作业：编辑距离，外卖小哥路径规划问题
## 课程5 自然语言处理初步1
#### Word2Vecter
#### Key word
#### NER
#### Dependency Parrsing
作业： 理解 Word2Vec 基本原理和提出背景，使用Gensim训练词向量，并进行同义词分析；词向量的T-SNE可视化
## 课程6 自然语言处理初步
#### 项目1 新闻人物言论自动提取

## 课程7 搜索引擎初步
#### 7.1文本与信息检索初步
#### 7.2 布尔搜索
#### 7.3 PageRank
作业：寻找专业性的论坛，快速制作网络爬虫，并通过倒排索引的方法构建一个搜索引擎。


## 课程8 机器学习初步
#### 8.1 机器学习的历史和发展原理
##### 机器学习的背景和原理
##### 机器学习的主要流派
##### 机器学习的现状分析
#### 8.2 过拟合和欠拟合
##### Bias 和 Variance 
##### 模型能力的分析
##### 数据能力的分析
##### 过拟合和欠拟合的原理和策略
#### 8.3 训练集、测试集、准确度
##### 数据对学习模型的影响
##### 训练集、测试集、准确度之间的关系
作业： 总结过拟合、欠拟合的原因；以及学习Scikit-learning;keras,tensorflow

## 课程9-11 经典机器学习模型
9.1 经典机器学习模型
- 9.1.1 回归和分类
- 9.1.2 Logstic Regression(逻辑回归)
- 9.1.3 KNN模型，
- 9.1.4 SVM
9.2 机器学习常见实践问题分析
- 9.2.1 天气预测
- 9.2.2 文本分类
- 9.2.3 图像分类
- 9.2.4 机器阅读理解
- 9.2.5 博弈问题
###  课程10-11内容：
+ SVM及核方法 
+ 贝叶斯方法
+ 决策树
+ XGBoost
+ 非监督学习
+ 项目代码分析
## 课程12 kmeans ,word2vec advanced
#### 0. Supplement: Why the Bayes we called Naive Bayes? 

#### 1. K-means & Hierarchy Cluster

#### 2. Embedding and Word2Vec Review

#### 3. The constraint of naive Word2Vec 

#### 4. Hierarchy Softmax and Negative Samples

#### 5. Other Word2Vec Methods:
+ Glove
+ Cove
+ EMLO
+ Bert
## 课程13 Neural Networks (神经网络)模型
#### 13.1 神经网络
+ 13.1.1 Loss函数，Backpropagation
+ 13.1.2 梯度下降
+ 13.1.3 softmax, cossentropy
+ 13.1.4 Optimizer 优化器
#### 13.2 神经网络的实践分析
+ 13.2.1 模型的稳定性
+ 13.2.2 模型的可解释性
+ 13.2.3 模型的运行分析
#### 13.3 实例分析：手动从零实现一个神经网络模型
+ 13.3.1 实现神经元
+ 13.3.2 实现拓扑排序
+ 13.3.3 实现 Backpropagation
+ 13.3.4 实现神经元权重自动调整
+ 13.3.5 利用完成的神经网络模型进行真实机器学习任务
#### homework :
1 tensorflow 代码实战（udacity,hands on tf;极客时间）
2.手动实现一个神经网络
3. 基于keras 实现CNN网络,并完成对MNIST数据的分类操作

## 第14课 CNN卷积神经网络

引言. 项目2的介绍

1 卷积神经网络与 Spatial Invariant
    1.1 卷积神经网络的历史背景
    1.2 卷积神经网络空间平移不变形(Spatial Invariant)的原理
    1.3 卷积神经网络与 weights sharing
    1.4 卷积神经网络的原理及Python 实现

2 Pooling, Dropout 与 Batch Normalization
     2.1 Pooling
     2.2 Dropout
     2.3 Batch Normalization 

4 CNN 的可视化
5 经典 CNN 模型分析：
    5.1 LeNet
    5.2 AlexNet
    5.3 GoogLeNet
    5.4 ResNet
    5.5 DenseNet

6 Transfer Learning 迁移学习
     6.1 迁移学习的背景
     6.2 迁移学习的方法
     6.3 Python 实现迁移学习的最佳实践
     
###  课程15 RNN循环网络 

1. 序列模型的提出背景；
2. 循环网络（RNN）的提出基本原理；
3. 梯度爆炸和弥散（exploding & vanish）
3. LSTM和GRU
4. tensorflow和keras中RNN的使用
5. Transfer Learning
