import pandas as pd
import seaborn as sns
from matplotlib.pyplot import show
import matplotlib.pyplot as plt
from mpl_toolkits import  mplot3d
import numpy as np
import tensorflow as tf


sns.set(style='whitegrid',palette='dark')

#获取数据
df0 = pd.read_csv('data_0.csv',names=['square', 'price'])
#print(df0)
#df0['square'] = df0['square'].astype('int')
# sns.lmplot('square','price',df0,height=6,fit_reg=True)
# show()

#多变量房价预测
df1 = pd.read_csv('data1.csv',names=['square', 'bedrooms', 'price'])

figure  = plt.figure()
#创建一个Axes 3d object
# ax = plt.axes(projection = '3d')
# #设置三个轴的坐标名称
# ax.set_xlabel('square')
# ax.set_ylabel('bedrooms')
# ax.set_zlabel('price')
# 绘制3D 散点图
# ax.scatter3D(df1['square'],df1['bedrooms'],df1['price'], c=df1['price'],cmap='Reds')
# show()

#数据规范化
def normalize_feature(df):
    return df.apply(lambda column: (column - column.mean())/column.std())

df = normalize_feature(df1)
#添加ones列 （x0）
ones = pd.DataFrame({'ones':np.ones(len(df))})
df = pd.concat([ones,df],axis = 1)

# ax = plt.axes(projection='3d')
# ax.set_xlabel('square')
# ax.set_ylabel('bedrooms')
# ax.set_zlabel('price')
# ax.scatter3D(df['square'], df['bedrooms'], df['price'], c=df['price'], cmap='Reds')
# show()


##训练模型
X_data = np.array(df[df.columns[0:3]])
y_data = np.array(df[df.columns[-1]]).reshape(len(df),1)
# print(X_data.shape,type(X_data))
# print(y_data.shape,type(y_data))

#创建线性回归模型（数据流图）
alpha = 0.01#学习率参数
epoch = 500#训练全量数据集的轮数

#输入X，47x3
X = tf.placeholder(tf.float32,X_data.shape)
#输入y，47x1
y = tf.placeholder(tf.float32,y_data.shape)
#权重变量 W，形状[3,1]
W = tf.get_variable('weights',(X_data.shape[1],1),initializer=tf.constant_initializer())
# 假设函数 h(x) = w0*x0+w1*x1+w2*x2, 其中x0恒为1
# 推理值 y_pred  形状[47,1]
y_pred = tf.matmul(X, W)
# 损失函数采用最小二乘法，y_pred - y 是形如[47, 1]的向量。
# tf.matmul(a,b,transpose_a=True) 表示：矩阵a的转置乘矩阵b，即 [1,47] X [47,1]
# 损失函数操作 loss
loss_op = 1/(2*len(X_data))*tf.matmul((y_pred-y),(y_pred-y),transpose_a=True)
#随机梯度下降优化器 opt
opt = tf.train.GradientDescentOptimizer(learning_rate=alpha)
# 单轮训练操作 train_op
train_op = opt.minimize(loss_op)


##创建会话（运行环境）
with tf.Session() as sess:
    #初始化全局变量
    sess.run(tf.global_variables_initializer())
    #开始训练模型
    # 因为训练量较小，所以使用全量数据量训练
    loss_data = []
    for e in range(1,epoch+1):
        sess.run(train_op,feed_dict={X:X_data,y:y_data})
        # 记录每一轮损失值变化情况
        loss, w = sess.run([loss_op, W], feed_dict={X: X_data, y: y_data})
        loss_data.append(float(loss))
        if e%10==0:
            res_str = 'Epoch %d \t Loss = %.4g\t  Model: y = %.4gx1 + %.4gx2 + %.4g'
            print(res_str %(e, loss, w[1], w[2], w[0]))

#可视化 loss
sns.set( style="whitegrid", palette="dark")
ax = sns.lineplot(x='epoch', y='loss', data=pd.DataFrame({'loss': loss_data, 'epoch': np.arange(epoch)}))
ax.set_xlabel('epoch')
ax.set_ylabel('loss')
plt.show()