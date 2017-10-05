import numpy as np
 
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils


nb_classes = 10
 
(X_train, y_train), (X_test, y_test) = mnist.load_data()
 
X_train = X_train.reshape(60000, 784).astype('float32') / 255
X_test  = X_test.reshape(10000, 784).astype('float32')  / 255
 
y_train = np_utils.to_categorical(y_train, nb_classes)
y_test  = np_utils.to_categorical(y_test,  nb_classes)
 
batch_size = 128
nb_epoch   = 10
 
model = Sequential()
model.add(Dense(512, input_shape=(784,)))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(10))
model.add(Activation('softmax'))
 
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=["accuracy"])
 
model.fit(X_train, y_train, nb_epoch=nb_epoch)
 
score = model.evaluate(X_test, y_test, verbose=0)
print('Loss Test score:',    score[0])
print('Accuracy Test score:',    score[1])


