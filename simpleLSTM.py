from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout, Highway
from keras.layers.recurrent import LSTM

data_dim = 61
layer_count = 4
dropout = 0.04
hidden_units = 61
nb_epoch = 10

model = Sequential()
model.add(Dense(hidden_units, input_dim=data_dim))
model.add(Dropout(dropout))
for index in range(layer_count):
    model.add(Highway(init='orthogonal', activation = 'relu'))
    model.add(Dropout(dropout))
model.add(Dropout(dropout))
model.add(Dense(61, activation='sigmoid'))


print 'compiling...'
model.compile(loss='binary_crossentropy', optimizer='adagrad')
model.fit(X, Y, batch_size=10, nb_epoch=nb_epoch,
    show_accuracy=True, validation_data=(x_test.T, y_test.T), shuffle=True, verbose=0)

predictions = model.predict_proba(x_test.T)