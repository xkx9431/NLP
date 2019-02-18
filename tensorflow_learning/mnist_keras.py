# tf_keras_learning
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

minst = input_data.read_data_sets('./mnist/dataset/')

# 加载 mnist 数据集

from keras.datasets import mnist
(x_train,y_train),(x_test,y_test) = mnist.load_data(path='./mnist/mnist.npz')
print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

import matplotlib.pylab as plt
fig = plt.figure()
for i in range(15):
    plt.subplot(3,5,i+1)
    plt.tight_layout()#自动适配子图尺寸
    plt.imshow(x_train[i],cmap='Greys')
    plt.title('Lable :{}'.format(y_train[i]))
    plt.xticks([])
    plt.yticks([])
plt.show()
