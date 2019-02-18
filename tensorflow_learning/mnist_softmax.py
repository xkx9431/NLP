from keras.datasets import mnist
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

#统计训练数据中各标签的数量
import  numpy as np
import  matplotlib.pylab as plt
label,count = np.unique(y_train,return_counts=True)
print(label,count)

# fig = plt.figure()
# plt.bar(label,count,width = 0.7,align = 'center')
# plt.title('Lable Distribution')
# plt.xlabel('LABELS')
# plt.ylabel('Count')
# plt.xticks(label)
# plt.ylim(0,7500)
#
# for a,b in zip(label,count):
#     plt.text(a,b,'%d'% b ,ha = 'center',va = 'bottom',fontsize=10)
# plt.show()

#
from keras.utils import np_utils
n_class = 10
print('shape before one-hot encoding',y_train.shape)
Y_train =  np_utils.to_categorical(y_train,n_class)
print('shape after one-hot encoding',Y_train.shape )
Y_test = np_utils.to_categorical(y_test, n_class)
#
print(y_train[0])
print(Y_train[0])

#使用 Keras sequential model 定义神经网络
from keras.models import Sequential
from keras.layers.core import Dense,Activation

model  = Sequential()
model.add(Dense(512,input_shape=(784,)))
model.add(Activation('relu'))

model.add(Dense(512))
model.add(Activation('relu'))

model.add(Dense(10))
model.add(Activation('softmax'))

#编译模型

model.compile(loss = 'categorical_crossentropy',optimizer='adam',metrics = ['accuracy'],)


#开始训练模型，并讲模型保存到history 中
history = model.fit(X_train,Y_train,batch_size=128,epochs=5,verbose=2,validation_data=(X_test,Y_test))

#可视化 训练数据
fig= plt.figure()
plt.subplot(2,1,1)
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend(['train','test'],loc ='lower right')

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
save_dir = './mnist_model/'
if gfile.Exists(save_dir):
    gfile.DeleteRecursively(save_dir)
gfile.MakeDirs(save_dir)

model_name = 'keras_mnist,h5'
model_path = os.path.join(save_dir,model_name)
model.save(model_path)
print('saved trained model at %s ' % model_path)



