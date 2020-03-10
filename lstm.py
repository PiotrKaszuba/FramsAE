import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from keras.layers import *
from keras.models import Model, Sequential
from encodeGenos import testEncode, inverse
from keras.preprocessing.sequence import pad_sequences
from keras.optimizers import Adam
sequences, encoders = testEncode()

import tensorflow as tf
sequence = pad_sequences(sequences, padding='post', dtype='float32', value=0)
max_len = np.shape(sequence)[1]
batch = np.shape(sequence)[0]
#sequence = sequence[:, :50, :]
max_len = sequence.shape[1]
features = sequence.shape[2]

zeros = np.zeros(shape=(batch, 32))

twoLstmLayers = False
useInitialState = True
inp = Input(shape=(max_len, features))
inp2 = Input(shape=[32])
#mask = Masking(0.001, input_shape=(max_len, features))(inp)
#embed = Dense(8, activation='linear', use_bias=False, kernel_initializer='he_normal')(inp)

def oneLayer(prevLayer, inp2, useInitialState=True):
    encoder3, h, c = LSTM(32, activation='relu', return_sequences=False, kernel_initializer='he_normal',
                          return_state=True)(prevLayer)
    decoder1 = RepeatVector(max_len)(inp2)
    if useInitialState:
        decoder2 = LSTM(32, activation='relu', return_sequences=True, kernel_initializer='he_normal')(decoder1,
                                                                                                  initial_state=[h, c])
    else:
        decoder2 = LSTM(32, activation='relu', return_sequences=True, kernel_initializer='he_normal')(decoder1)
    return decoder2


def twoLayers(prevLayer, inp2, useInitialState=True):
    encoder2 = LSTM(32, activation='relu', return_sequences=True, kernel_initializer='he_normal')(prevLayer)
    encoder3, h, c = LSTM(32, activation='relu', return_sequences=False, kernel_initializer='he_normal',
                          return_state=True)(encoder2)
    # define reconstruct decoder
    decoder1 = RepeatVector(max_len)(inp2)
    if useInitialState:
        decoder2 = LSTM(32, activation='relu', return_sequences=True, kernel_initializer='he_normal')(decoder1,
                                                                                                      initial_state=[h,
                                                                                                                     c])
    else:
        decoder2 = LSTM(32, activation='relu', return_sequences=True, kernel_initializer='he_normal')(decoder1)
    decoder3 = LSTM(32, activation='relu', return_sequences=True, kernel_initializer='he_normal')(decoder2)
    return decoder3


if twoLstmLayers:
    decoder3 = twoLayers(inp, inp2, useInitialState)
else:
    decoder3 = oneLayer(inp, inp2, useInitialState)

decoder4 = TimeDistributed(Dense(features, activation='softmax', kernel_initializer='he_normal'))(decoder3)

model = Model(inputs=[inp, inp2], outputs=decoder4)
model.summary()

model.compile(optimizer=Adam(0.005), loss='binary_crossentropy')
history = model.fit([sequence,zeros], sequence,
                      epochs=30,
                      verbose=2,
                      validation_split=0.2,
                      shuffle=True)

res = model.predict([sequence,zeros])



# get sequences with ones in maxes
argmax = np.argmax(res, axis=-1)
ind = np.indices(argmax.shape)
argmax = np.expand_dims(argmax, axis=0)
ind = np.concatenate([ind, argmax], axis =0)
row, col, argm = ind
resMax = np.zeros_like(res)
resMax[row, col, argm] = 1
#

# get genotype for predicted
genos = inverse(resMax, encoders)
# get genotype for original
gen2 = inverse(sequence, encoders)

print("genos now")
for i in range(len(genos)):

    print("geno " + str(i))
    print("-----original-------")
    print(gen2[i])
    print("-----predicted------")
    print(genos[i])

import matplotlib.pyplot as plt
import pickle
with open('historyOneLayer', 'wb') as handle:
    pickle.dump(history.history['val_loss'], handle)
plt.plot(list(range(len(history.history['val_loss']))), history.history['val_loss'])
plt.show()