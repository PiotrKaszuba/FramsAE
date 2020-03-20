import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from keras.layers import *
import keras.backend as K
from keras.models import Model, Sequential
from encodeGenos import testEncode, inverse, oneHot, prepareEncoders, encodeGenos
from keras.preprocessing.sequence import pad_sequences
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras.models import load_model
import pickle
from keras.regularizers import l1_l2
from ParseGenFile import dict,testGenos

import tensorflow as tf

genos = testGenos()
more_genos = testGenos("customGens2.gen")
encoders = prepareEncoders(dict)

sequences = encodeGenos(genos, encoders, oneHot)
sequences2 = encodeGenos(genos, encoders, True)

features = len(dict)

X = pad_sequences(sequences, padding='pre', dtype='int32', value=-1)
X+=1
#X = np.expand_dims(X,-1)
Y = pad_sequences(sequences2, padding='pre', dtype='float32', value=0.0)
Y = np.flip(Y, 1)#pad_sequences(sequences, padding='post', dtype='float32', value=0)


zerosInputs = np.where(X==0, 0, 1)
zerosInputs = np.flip(zerosInputs, 1)


X_train, X_test, y_train, y_test, zerosTrain, zerosTest = train_test_split(X, Y, zerosInputs, test_size=0.2, random_state=42)

max_len = np.shape(X)[1]
batch = np.shape(X)[0]

# more_sequences = encodeGenos(genos, encoders, oneHot)
# more_sequences2 = encodeGenos(genos, encoders, True)
# more_sequences = [x for x in more_sequences if np.shape(x)[0] <= max_len]
# more_sequences2 = [x for x in more_sequences2 if np.shape(x)[0] <= max_len]
#
# X_more = pad_sequences(more_sequences, padding='pre', maxlen=max_len, dtype='int32', value=-1)
# X_more+=1
# Y_more = pad_sequences(more_sequences2, padding='pre', maxlen=max_len, dtype='float32', value=0.0)
# Y_more = np.flip(Y_more, 1)
#
# zerosInputsMore = np.where(X_more==0, 0, 1)
# zerosInputsMore = np.flip(zerosInputsMore, 1)
#
# X_train = np.concatenate([X_train, X_more], axis=0)
# zerosTrain = np.concatenate([zerosTrain, zerosInputsMore], axis=0)
# y_train = np.concatenate([y_train, Y_more], axis=0)

#sequence = sequence[:, :50, :]
#features = X.shape[2]





# zerosTrain = np.zeros(shape=(np.shape(X_train)[0], 32))
# zerosTest = np.zeros(shape=(np.shape(X_test)[0], 32))

bidir= True
twoLstmLayers = True
useInitialState = True

inp = Input(shape=(max_len,))
inp2 = Input(shape=(max_len,))
#mask = Masking(features, input_shape=(max_len,))(inp)
#embed = Dense(8, activation='linear', use_bias=False, kernel_initializer='he_normal')(inp)
embedLayer = Embedding(features+1, 6, input_length=max_len, mask_zero=True, name='inpEmbed1')
embed=embedLayer(inp)
embed2Layer = Embedding(2, 32, input_length=max_len, mask_zero=True, trainable=False, name='inpEmbed2')
embed2 = embed2Layer(inp2)
#print(embed2Layer.get_weights()[0][0])
embed2Layer.set_weights(np.expand_dims(np.stack([np.zeros_like(embed2Layer.get_weights()[0][0]), np.ones_like(embed2Layer.get_weights()[0][0])], axis=0), axis=0))
prev_layer = embed
prev_layer2 = embed2
def oneLayer(prevLayer, inp2, useInitialState=True, bidir = False):
    if not bidir:
        enc, h, c = LSTM(32, activation='relu', return_sequences=False, kernel_initializer='he_normal',
                               return_state=True)(prevLayer)
    else:
        enc, h1,h2,h3,h4 = Bidirectional(LSTM(32, activation='relu', return_sequences=False, kernel_initializer='he_normal',
                               return_state=True))(prevLayer)


        concat = Concatenate()([h1,h2,h3,h4])
        hLayer = Dense(32, activation='sigmoid', use_bias=True, kernel_initializer='uniform')
        h = hLayer(concat)
        #hLayer.set_weights(np.zeros_like(hLayer.get_weights()))
        cLayer = Dense(32, activation='linear', use_bias=False, kernel_initializer='uniform')
        c = cLayer(concat)
        #cLayer.set_weights(np.zeros_like(cLayer.get_weights()))

    #decoder1 = RepeatVector(max_len)(inp2)
    if useInitialState:
        decoder2Layer = LSTM(32, activation='relu', return_sequences=True, kernel_initializer='he_normal')
        decoder2= decoder2Layer(inp2, initial_state=[h,c])
    else:
        decoder2 = LSTM(32, activation='relu', return_sequences=True, kernel_initializer='he_normal')(inp2)
    return decoder2


def twoLayers(prevLayer, inp2, useInitialState=True, bidir=False):


    if not bidir:
        encoder2 = LSTM(32, activation='relu', return_sequences=True, kernel_initializer='he_normal')(prevLayer)
        enc, h, c = LSTM(32, activation='relu', return_sequences=False, kernel_initializer='he_normal',
                               return_state=True)(encoder2)
    else:
        encoder2 = Bidirectional(LSTM(32, activation='relu', name='encoderBidir1', return_sequences=True, kernel_initializer='he_normal', kernel_regularizer=l1_l2(0, 0), bias_regularizer=l1_l2(0, 0)),name='encoderBidir1')(prevLayer)
        enc, h1,h2,h3,h4 = Bidirectional(LSTM(32, activation='relu', name='encoderBidir2',return_sequences=False, kernel_initializer='he_normal', recurrent_regularizer=l1_l2(0, 0.0), kernel_regularizer=l1_l2(0, 0), bias_regularizer=l1_l2(0, 0),
                               return_state=True), name='encoderBidir2')(encoder2)


        concat = Concatenate()([h1,h2,h3,h4])
        bn1 = BatchNormalization(name='bn1')(concat)
        hLayer = Dense(32, activation='tanh', use_bias=True, kernel_initializer='he_normal', name='hDense', bias_regularizer=l1_l2(5e-6, 5e-6))
        h = hLayer(bn1)
        #hLayer.set_weights(np.zeros_like(hLayer.get_weights()))
        cLayer = Dense(32, activation='linear', use_bias=True, kernel_initializer='he_normal', name='cDense', activity_regularizer=l1_l2(2e-7,0.0))
        c = cLayer(bn1)
        # cLayer.set_weights(np.zeros_like(cLayer.get_weights()))



    # define reconstruct decoder
    #decoder1 = RepeatVector(max_len)(inp2)

    decoderInpH = Input((32,))
    decoderInpC = Input((32,))
    decoderPrevInput = Input(((max_len,32)))
    if useInitialState:
        rep = RepeatVector(max_len)(decoderInpH)
        mult = Multiply()([decoderPrevInput,rep])
        mask = Masking(0.0)(mult)
        decoder2 = LSTM(32, activation='relu', return_sequences=True, kernel_initializer='he_normal', name='decoder1', kernel_regularizer=l1_l2(0, 0), bias_regularizer=l1_l2(0, 0))(mask,
                                                                                                      initial_state=[decoderInpH,
                                                                                                                     decoderInpC])
    else:
        decoder2 = LSTM(32, activation='relu', return_sequences=True, kernel_initializer='he_normal')(inp2)
    decoder3 = LSTM(32, activation='relu', return_sequences=True, kernel_initializer='he_normal', name='decoder2', kernel_regularizer=l1_l2(0, 0), bias_regularizer=l1_l2(0,0))(decoder2)



    return decoder3, h, c, decoderInpH, decoderInpC, decoderPrevInput


if twoLstmLayers:
    decoder3,h,c, decoderInpH, decoderInpC,decoderPrevInput = twoLayers(prev_layer, prev_layer2, useInitialState, bidir)
else:
    decoder3,h,c, decoderInpH, decoderInpC,decoderPrevInput = oneLayer(prev_layer, prev_layer2, useInitialState, bidir)

decoder4 = TimeDistributed(Dense(features, activation='softmax', kernel_initializer='he_normal', name='denseOut'), name='denseOut')(decoder3)

decoder = Model(inputs=[decoderInpH,decoderInpC,decoderPrevInput], outputs=decoder4)

model = Model(inputs=[inp, inp2], outputs=decoder([h,c,prev_layer2]))
model.summary()
decoder.summary()
model.compile(optimizer=Adam(0.000002), loss='categorical_crossentropy')
decoder.compile(optimizer=Adam(0.000002), loss='categorical_crossentropy')

layer_outputh = model.get_layer('hDense').output
layer_outputc = model.get_layer('cDense').output

encoder_model = Model(inputs=[inp, inp2], outputs=[layer_outputh, layer_outputc, prev_layer2])


model.load_weights('models/modelBiDir22509_0.9718959477721824_weights_spare', by_name=True)
decoder.load_weights('models/modelBiDir22509_0.9718959477721824_weights_spare', by_name=True)
#model = load_model('models/modelBiDir22410_0.9691729474965206')
# model.save_weights('models/modelBiDir22509_0.9718959477721824_weights_spare')

zerosTestRead = np.ones_like(zerosTest)
zerosTestReadCheck = np.flip(zerosTestRead, 1)

outH, outC, prevL2 = encoder_model.predict([X_test, zerosTestRead])
dec = decoder.predict([outH, outC, prevL2])



# get genotype for original
X_testCheck = X_test-1
X_testCheck = np.where(X_testCheck == -1, 0, X_testCheck)
gen2Check = inverse(X_testCheck, encoders, False)
gen2Check = np.array(gen2Check)

zerosTestCheck = np.flip(zerosTest, 1)


# get sequences with ones in maxes
argmax = np.argmax(dec, axis=-1)
ind = np.indices(argmax.shape)
argmax = np.expand_dims(argmax, axis=0)
ind = np.concatenate([ind, argmax], axis=0)
row, col, argm = ind
resMax = np.zeros_like(dec)
resMax[row, col, argm] = 1
#

# get genotype for predicted
genosCheck = inverse(resMax, encoders, True)

genosCheck = np.flip(genosCheck, 1)

genosCheck = np.array(genosCheck)

sumOfTokens = np.sum(zerosTestCheck)

sumOfHits = np.where((zerosTestCheck == 1) & (gen2Check == genosCheck), 1, 0).sum()

acc = sumOfHits / sumOfTokens
print("Token accuracy: " + str(acc))




def evaluate(model, testData, gen2Check, zerosTestCheck):
    res = model.predict(testData, batch_size=8192)
    genosCheck = []

    # get sequences with ones in maxes
    argmax = np.argmax(res, axis=-1)
    ind = np.indices(argmax.shape)
    argmax = np.expand_dims(argmax, axis=0)
    ind = np.concatenate([ind, argmax], axis=0)
    row, col, argm = ind
    resMax = np.zeros_like(res)
    resMax[row, col, argm] = 1
    #

    # get genotype for predicted
    genosCheck = inverse(resMax, encoders, True)

    genosCheck = np.flip(genosCheck, 1)

    genosCheck = np.array(genosCheck)

    sumOfTokens = np.sum(zerosTestCheck)

    sumOfHits = np.where((zerosTestCheck == 1) & (gen2Check == genosCheck), 1, 0).sum()

    acc = sumOfHits / sumOfTokens
    print("Token accuracy: " + str(acc))

    return genosCheck, acc


#
# for i in range(1000):
#
#
#     history = model.fit([X_train,zerosTrain], y_train,
#                           epochs=1,
#                           verbose=2,
#                           validation_data=([X_test, zerosTest], y_test),
#                           shuffle=True, batch_size=9046)
#
#     _, acc = evaluate(model, [X_test, zerosTest], gen2Check, zerosTestCheck)
#
#     model.save('models/modelBiDir'+str(i+22510 ) + '_' + str(acc), True, True)
#     model.save_weights('models/modelBiDir'+str(i+22510 ) + '_' + str(acc) + '_weights')
#     with open('models/modelBiDirHistory'+str(i+22510), 'wb') as handle:
#         pickle.dump(history.history['val_loss'], handle)
#
# print(embed2Layer.get_weights())

import matplotlib.pyplot as plt
#
#
# plt.plot(list(range(len(history.history['val_loss']))), history.history['val_loss'])
# plt.show()

# w = embedLayer.get_weights()[0]
#
# from scipy.spatial.distance import cdist
#
# dat = cdist(w,w, 'euclidean')
# print()
#
# f = plt.figure(figsize=(19, 15))
# ax = f.add_subplot(111)
# cax = ax.matshow(dat)
# # plt.xticks(range(np.shape(dat)[0]), np.shape(dat)[0], fontsize=14, rotation=45)
# # plt.yticks(range(np.shape(dat)[0]), np.shape(dat)[0], fontsize=14)
# ax.set_xticks(np.arange(21))
# ax.set_yticks(np.arange(21))
# ax.set_xticklabels(['']+ encoders[0].classes_.tolist())
# ax.set_yticklabels(['']+ encoders[0].classes_.tolist())
# # ax.set_xticklabels(['']+list(map(str, list(range(21)))))
# # ax.set_yticklabels(['']+list(map(str, list(range(21)))))
# plt.colorbar(cax, ax=ax)
# #cb.ax.tick_params(labelsize=14)
# plt.title('Similarity', fontsize=16)
# plt.show()

# genosCheck, acc = evaluate(model, [X_test, zerosTest], gen2Check, zerosTestCheck)




# for i in range(len(genos)):
#     gen2[i] = np.delete(gen2[i], np.where(zerosTest[i]==0))
#     genos[i] = np.delete(genos[i], np.where(zerosTest[i]==0))
print("genos now")
for i in range(200):

    print("geno " + str(i))
    print("-----original-------")
    print(np.delete(gen2Check[i], np.where(zerosTestCheck[i]==0)))
    print("-----predicted------")
    #print(genosCheck[i])
    print(np.delete(genosCheck[i], np.where(zerosTestCheck[i]==0)))
    print("-----encoded--------")
    print([outH[i], outC[i]])
    print("-----predicted FULL------")
    # print(genosCheck[i])
    print(np.delete(genosCheck[i], np.where(zerosTestReadCheck[i] == 0)))

print("Token accuracy: " + str(acc))
#
# model.summary()

import copy

g=0
hiddenH= copy.deepcopy(outH[g])
hiddenC= copy.deepcopy(outC[g])
print([hiddenH, hiddenC])
while(1):
    t = input()
    if t == 'geno':
        t = input()
        g = int(t)
        hiddenH = copy.deepcopy(outH[g])
        hiddenC = copy.deepcopy(outC[g])
        print("change to " + str(g) + " geno")
    elif t== 'rand':
        hiddenH += np.random.rand(32) - 0.5
        hiddenC += np.random.rand(32) - 0.5
    else:
        n = int(t)
        if n > 31:
            n-=32
            oper = hiddenC
            s = 'cell state '
        else:
            oper = hiddenH
            s = 'hidden state '
        f = float(input())
        print("changed " + s + " dimension "  +str(n) + " from " + str(oper[n]) + " to " + str(oper[n] + f))
        oper[n]+=f





    dec = decoder.predict([np.expand_dims(hiddenH, axis=0), np.expand_dims(hiddenC, axis=0), np.ones(shape=(1,44,32))])
    # get sequences with ones in maxes
    argmax = np.argmax(dec, axis=-1)
    ind = np.indices(argmax.shape)
    argmax = np.expand_dims(argmax, axis=0)
    ind = np.concatenate([ind, argmax], axis=0)
    row, col, argm = ind
    resMax = np.zeros_like(dec)
    resMax[row, col, argm] = 1
    #

    # get genotype for predicted
    genosCheck = inverse(resMax, encoders, True)

    genosCheck = np.flip(genosCheck, 1)

    genosCheck = np.array(genosCheck)

    print("-----original-------")
    print(np.delete(gen2Check[g], np.where(zerosTestCheck[g] == 0)))
    print("-----predicted------")
    print(np.delete(genosCheck[0], np.where(zerosTestCheck[g] == 0)))
    print("----------prediction full-----------------")
    print(genosCheck[0])
    print("------------current------------------")
    print([hiddenH, hiddenC])

