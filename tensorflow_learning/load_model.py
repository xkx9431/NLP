from keras.models import load_model
from keras.datasets import mnist
from keras.utils import np_utils
import numpy as np
(x_train,y_train),(x_test,y_test) = mnist.load_data('mnist/mnist.npz')
#
print(x_train.shape,y_train.shape,type(x_train),type(y_train))

#数据规范化
X_train = x_train.reshape(60000,28*28)
X_test = x_test.reshape(10000,28*28)
print(X_train.shape,type(X_train))

#将数据转换为 float 32
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
#数据归一化
X_train/=255
X_test/=255
#
n_class = 10
print('shape before one-hot encoding',y_train.shape)
Y_train = np_utils.to_categorical(y_train,n_class)
print('shape after one-hot encoding',Y_train.shape )
Y_test = np_utils.to_categorical(y_test, n_class)
model_path = 'mnist_model/keras_mnist,h5'
mnist_model = load_model(model_path)
#统计模型 在测试集上面的分类结果
loss_and_metrics  = mnist_model.evaluate(X_test, Y_test, verbose=2)
print('test loss :{}'.format(loss_and_metrics[0]))
print('test accuracy :{}'.format(loss_and_metrics[1]*100))

predicted_classes = mnist_model.predict_classes(X_test)

correct_indices = np.nonzero(predicted_classes == y_test)[0]
incorrect_indices = np.nonzero(predicted_classes != y_test)[0]
print("Classified correctly count: {}".format(len(correct_indices)))
print('5 correct indices',correct_indices[:5])
print("Classified incorrectly count: {}".format(len(incorrect_indices)))
print('5 incorrect indices',incorrect_indices[:5])