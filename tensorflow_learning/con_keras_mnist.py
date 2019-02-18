#加载MNIST 数据集
from keras.datasets import mnist
(x_train,y_train),(x_test,y_test) = mnist.load_data('mnist/mnist.npz')
print(x_train.shape,type(x_train))
print(y_train.shape,type(y_train))

#
from keras import backend as K
img_rows,img_cols = 28,28

if K.image_data_format()=='channel_first':
    x_train = x_train.reshape(x_train.shape[0],1,img_rows,img_cols)
    x_test = x_test.reshape(x_test.shape[0],img_rows,img_cols)
    input_shape = (1,img_rows,img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)
print(x_train.shape,type(x_train))
print(x_test.shape,type(x_test))



# 将数据类型转换为float32
X_train = x_train.astype('float32')
X_test = x_test.astype('float32')
# 数据归一化
X_train /= 255
X_test /= 255

print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# 统计训练数据中各标签数量
import numpy as np
import matplotlib.pyplot as plt
label,count  = np.unique(y_train,return_counts=True)
print(label,count)

fig  = plt.figure()
plt.bar(label,count,width = 0.7,align = 'center')
plt.title('label distribution')
plt.xlabel('label ')
plt.ylabel('count')
plt.xticks()
plt.ylim(0,7500)
for a,b in zip(label,count):
    plt.text(a,b,'%d'%b,ha = 'center',va = 'bottom',fontsize = 10)
plt.show()
#数据处理：one-hot 编码
from keras.utils import np_utils
n_classes = 10
print('shape before one-hot encoding:',y_train.shape)
Y_train = np_utils.to_categorical(y_train,n_classes)
print('after onr-hot encoding:',Y_train.shape)
Y_test = np_utils.to_categorical(y_test,n_classes)

print(y_train[0])
print(Y_train[0])

# 使用keras sequential model 定义mnist CNN 网络
from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten
from keras.layers import Conv2D,MaxPooling2D

model = Sequential()
## Feature Extraction
# 第1层卷积，32个3x3的卷积核 ，激活函数使用 relu
model.add(Conv2D(filters=32,kernel_size=(3,3),
                 activation='relu',input_shape=input_shape))
# 第2层卷积，64个3x3的卷积核，激活函数使用 relu
model.add(Conv2D(filters=64,kernel_size=(3,3),activation='relu'))

#最大池化层,池化窗口（2,2）
model.add(MaxPooling2D(pool_size=(2,2)))
#dropout 25%
model.add(Dropout(0.25))
# 将 Pooled feature map 摊平后输入全连接网络
model.add(Flatten())

##CLASSFICTION
#全连接层
model.add(Dense(128,activation='relu'))
model.add(Dropout(0.25))

#使用softmax 激活函数做多分类
model.add(Dense(n_classes,activation='softmax'))
print(model.summary())

for layer in model.layers:
    print(layer.get_output_at(0).get_shape().as_list())

#编译模型
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

#训练模型
history = model.fit(X_train,
                    Y_train,
                    batch_size=128,
                    epochs=5,
                    verbose=2,
                    validation_data=(X_test, Y_test))
#可视化指标
fig = plt.figure()
plt.subplot(2,1,1)
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='lower right')

plt.subplot(2,1,2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.tight_layout()

plt.show()

#保存模型
import os
import tensorflow.gfile as gfile

save_dir = "./mnist/model/"

if gfile.Exists(save_dir):
    gfile.DeleteRecursively(save_dir)
gfile.MakeDirs(save_dir)

model_name = 'cnn_keras_mnist.h5'
model_path = os.path.join(save_dir, model_name)
model.save(model_path)
print('Saved trained model at %s ' % model_path)

#加载模型
from keras.models import load_model

mnist_model = load_model(model_path)

#统计模型在测试集上面的分类结果
loss_and_metrics = mnist_model.evaluate(X_test,Y_test,verbose = 2)
print("Test Loss: {}".format(loss_and_metrics[0]))
print("Test Accuracy: {}%".format(loss_and_metrics[1]*100))
predicted_classes = mnist_model.predict_classes(X_test)

correct_indices = np.nonzero(predicted_classes == y_test)[0]
incorrect_indices = np.nonzero(predicted_classes != y_test)[0]
print("Classified correctly count: {}".format(len(correct_indices)))
print("Classified incorrectly count: {}".format(len(incorrect_indices)))