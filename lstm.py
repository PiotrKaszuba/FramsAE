import os
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow.keras.layers import *

from tensorflow.keras.models import Model
from Code.Preparation.encodeGenos import inverse, prepareEncoders, encodeGenos
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split

import pickle
from tensorflow.keras.regularizers import l1_l2
from Code.Preparation.ParseGenFile import testGenos, parseFitness
from Code.Preparation.configuration import get_config
import numpy as np
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"
representation = 'f1'
oneHot = False
dict = 'XqQrRlLcCfFmM()iI,ST'
if representation=='f1':
    dimsEmbed = 6
if representation=='f4':
    dimsEmbed = 6
if representation=='f9':
    dimsEmbed = 4

model_name = 'test'
representation = 'f1'
long_genos = 'short'
cells = 48
twoLayer = 'oneLayer'
bidir = 'Bidir'
data_path = 'models/'
load_dir = ''
config = get_config(model_name, representation, long_genos, cells, twoLayer, bidir, data_path, load_dir)
fitness = parseFitness("ocenione" + representation+".gen")



genos = testGenos(config, print_some_genos=True)

#genos = testGenos(representation+"TestDissim.gen")
#more_genos = testGenos("customGens2.gen")
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


X_train, X_test, y_train, y_test, zerosTrain, zerosTest, fitnessTrain, fitnessTest = train_test_split(X, Y, zerosInputs, fitness, test_size=0.2, random_state=42)


# X_train, X_test, y_train, y_test, zerosTrain, zerosTest = train_test_split(X, Y, zerosInputs, test_size=0.2, random_state=42)
# zerosTest = zerosInputs
# X_test = X

max_len = np.shape(X)[1]
batch = np.shape(X)[0]

# more_sequences = encodeGenos(more_genos, encoders, oneHot)
# more_sequences2 = encodeGenos(more_genos, encoders, True)
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
lr = 0.0005
batch_size = 2048

past_epochs = 1
epochs_per_i = 100
model_name = 'modelOneLayerBidir'
loaded_model_acc = 0
bidir= True
twoLstmLayers = True
useInitialState = True

inp = Input(shape=(max_len,))
inp2 = Input(shape=(max_len,))
#mask = Masking(features, input_shape=(max_len,))(inp)
#embed = Dense(8, activation='linear', use_bias=False, kernel_initializer='he_normal')(inp)
embedLayer = Embedding(features+1, dimsEmbed, input_length=max_len, mask_zero=True, name='inpEmbed1')
embed=embedLayer(inp)
embed2Layer = Embedding(2, 32, input_length=max_len, mask_zero=True, trainable=False, name='inpEmbed2')
embed2 = embed2Layer(inp2)
#print(embed2Layer.get_weights()[0][0])
embed2Layer.set_weights(np.expand_dims(np.stack([np.zeros_like(embed2Layer.get_weights()[0][0]), np.ones_like(embed2Layer.get_weights()[0][0])], axis=0), axis=0))
prev_layer = embed
prev_layer2 = embed2
def oneLayer(prevLayer, inp2, useInitialState=True, bidir = False):
    if not bidir:
        enc, h1, c1 = LSTM(32, activation='relu', return_sequences=False, kernel_initializer='he_normal', name='encoder2',
                               return_state=True, recurrent_regularizer=l1_l2(1e-7, 1e-7), kernel_regularizer=l1_l2(2e-6, 2e-6), bias_regularizer=l1_l2(4e-6, 4e-6))(prevLayer)
        concat = Concatenate()([h1, c1])

    else:
        enc, h1, h2, h3, h4 = Bidirectional(
            LSTM(32, activation='relu', name='encoderBidir2', return_sequences=False, kernel_initializer='he_normal',
                 recurrent_regularizer=l1_l2(1e-7, 1e-7), kernel_regularizer=l1_l2(2e-6, 2e-6),
                 bias_regularizer=l1_l2(4e-6, 4e-6),
                 return_state=True), name='encoderBidir2')(prev_layer)


        concat = Concatenate()([h1,h2,h3,h4])

    bn1 = BatchNormalization(name='bn1')(concat)
    hLayer = Dense(32, activation='tanh', use_bias=True, kernel_initializer='he_normal', name='hDense',
                   activity_regularizer=l1_l2(1e-6, 1e-6), bias_regularizer=l1_l2(5e-6, 5e-6))
    h = hLayer(bn1)
    # hLayer.set_weights(np.zeros_like(hLayer.get_weights()))
    cLayer = Dense(32, activation='linear', use_bias=True, kernel_initializer='he_normal', name='cDense',
                   activity_regularizer=l1_l2(1e-6, 1e-6), bias_regularizer=l1_l2(5e-6, 5e-6))
    c = cLayer(bn1)

    decoderInpH = Input((32,))
    decoderInpC = Input((32,))
    decoderPrevInput = Input(((max_len, 32)))
    #decoder1 = RepeatVector(max_len)(inp2)
    if useInitialState:
        rep = RepeatVector(max_len)(decoderInpH)
        mult = Multiply()([decoderPrevInput, rep])
        mask = Masking(0.0)(mult)
        decoder2 = LSTM(32, activation='relu', return_sequences=True, kernel_initializer='he_normal', name='decoder1',
                        kernel_regularizer=l1_l2(3e-6, 3e-6), bias_regularizer=l1_l2(5e-6, 5e-6))(mask,
                                                                                                  initial_state=[
                                                                                                      decoderInpH,
                                                                                                      decoderInpC])

    else:
        decoder2 = LSTM(32, activation='relu', return_sequences=True, kernel_initializer='he_normal')(inp2)
    return decoder2, h, c, decoderInpH, decoderInpC, decoderPrevInput


def twoLayers(prevLayer, inp2, useInitialState=True, bidir=False):


    if not bidir:
        encoder2 = LSTM(32, activation='relu', return_sequences=True, kernel_initializer='he_normal')(prevLayer)
        enc, h1, c1 = LSTM(32, activation='relu', return_sequences=False, kernel_initializer='he_normal',
                               return_state=True, name='encoder2')(encoder2)
        concat = Concatenate()([h1, c1])
    else:
        encoder2 = Bidirectional(LSTM(32, activation='relu', name='encoderBidir1', return_sequences=True, kernel_initializer='he_normal', kernel_regularizer=l1_l2(2e-6, 2e-6), bias_regularizer=l1_l2(4e-6, 4e-6)),name='encoderBidir1')(prevLayer)
        enc, h1,h2,h3,h4 = Bidirectional(LSTM(32, activation='relu', name='encoderBidir2',return_sequences=False, kernel_initializer='he_normal', recurrent_regularizer=l1_l2(1e-7, 1e-7), kernel_regularizer=l1_l2(2e-6, 2e-6), bias_regularizer=l1_l2(4e-6, 4e-6),
                               return_state=True), name='encoderBidir2')(encoder2)


        concat = Concatenate()([h1,h2,h3,h4])
    bn1 = BatchNormalization(name='bn1')(concat)
    hLayer = Dense(32, activation='tanh', use_bias=True, kernel_initializer='he_normal', name='hDense', activity_regularizer=l1_l2(1e-6,1e-6), bias_regularizer=l1_l2(5e-6, 5e-6))
    h = hLayer(bn1)
    #hLayer.set_weights(np.zeros_like(hLayer.get_weights()))
    cLayer = Dense(32, activation='linear', use_bias=True, kernel_initializer='he_normal', name='cDense', activity_regularizer=l1_l2(1e-6,1e-6), bias_regularizer=l1_l2(5e-6, 5e-6))
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
        decoder2 = LSTM(32, activation='relu', return_sequences=True, kernel_initializer='he_normal', name='decoder1', kernel_regularizer=l1_l2(3e-6, 3e-6), bias_regularizer=l1_l2(5e-6, 5e-6))(mask,
                                                                                                      initial_state=[decoderInpH,
                                                                                                                     decoderInpC])
    else:
        decoder2 = LSTM(32, activation='relu', return_sequences=True, kernel_initializer='he_normal')(inp2)
    decoder3 = LSTM(32, activation='relu', return_sequences=True, kernel_initializer='he_normal', name='decoder2', kernel_regularizer=l1_l2(3e-6, 3e-6), bias_regularizer=l1_l2(5e-6,5e-6))(decoder2)



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
model.compile(optimizer=Adam(lr), loss='categorical_crossentropy')
decoder.compile(optimizer=Adam(lr), loss='categorical_crossentropy')
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
# plot_model(model, to_file='model.png')
# plot_model(decoder, to_file='decoder.png')
layer_outputh = model.get_layer('hDense').output
layer_outputc = model.get_layer('cDense').output


encoder_model = Model(inputs=[inp, inp2], outputs=[layer_outputh, layer_outputc, prev_layer2])

# comment
if past_epochs > 0:
    if bidir and twoLstmLayers:
        if representation == 'f1':
            model.load_weights('models/modelBiDir22509_0.9718959477721824_weights_spare', by_name=True)
            decoder.load_weights('models/modelBiDir22509_0.9718959477721824_weights_spare', by_name=True)
        if representation == 'f4':
            pass
            model.load_weights('models/modelBiDirf4_3780_0.7178693819185917_weights', by_name=True)
            decoder.load_weights('models/modelBiDirf4_3780_0.7178693819185917_weights', by_name=True)
        if representation == 'f9':
            model.load_weights('models/modelBiDirf9_8000_0.9873158029708243_weights', by_name=True)
            decoder.load_weights('models/modelBiDirf9_8000_0.9873158029708243_weights', by_name=True)

    if not bidir and not twoLstmLayers:
        if representation == 'f1':
            pass
            print("loaded")
            model.load_weights('models/modelOneLayerf1_12700_0.8167521666341699_weights', by_name=True)
            decoder.load_weights('models/modelOneLayerf1_12700_0.8167521666341699_weights', by_name=True)
            # model.load_weights('models/modelOneLayerf1_27300_0.8691816879912326_weights', by_name=True)
            # decoder.load_weights('models/modelOneLayerf1_27300_0.8691816879912326_weights', by_name=True)

    if bidir and not twoLstmLayers:
        if representation == 'f1':
            pass

            model.load_weights('models/modelOneLayerBidirf1_9000_0.8579871313024009_weights', by_name=True)
            decoder.load_weights('models/modelOneLayerBidirf1_9000_0.8579871313024009_weights', by_name=True)
            # model.load_weights('models/' + model_name + representation + '_' + str(past_epochs) + '_' + str(loaded_model_acc) + '_weights',by_name=True)
            # decoder.load_weights('models/'+model_name+representation+'_'+str(past_epochs) + '_' + str(loaded_model_acc) + '_weights', by_name=True)

#model.load_weights('models/'+model_name+representation+'_'+str(past_epochs) + '_' + str(loaded_model_acc) + '_weights', by_name=True)
#decoder.load_weights('models/'+model_name+representation+'_'+str(past_epochs) + '_' + str(loaded_model_acc) + '_weights', by_name=True)

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

for i in range(1000):
    history = model.fit([X_train,zerosTrain], y_train,
                          epochs=epochs_per_i,
                          verbose=1,
                          validation_data=([X_test, zerosTest], y_test),
                          shuffle=True, batch_size=batch_size)

    _, acc = evaluate(model, [X_test, zerosTest], gen2Check, zerosTestCheck)

    model.save('models/'+model_name+representation+'_'+str((i+1)*epochs_per_i + past_epochs) + '_' + str(acc), True, True)
    model.save_weights('models/'+model_name+representation+'_'+str((i+1)*epochs_per_i + past_epochs) + '_' + str(acc) + '_weights')
    with open('models/'+model_name+representation+'_History'+str((i+1)*epochs_per_i + past_epochs), 'wb') as handle:
        pickle.dump(history.history['val_loss'], handle)
# #

# print(embed2Layer.get_weights())

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


# ## fitness - distance
howMany=8000
inner = np.concatenate([outH, outC], axis=-1)[:howMany]
fitnessTest = np.expand_dims(fitnessTest, axis=-1)[:howMany]
from scipy.spatial.distance import cdist

dists = cdist(inner, inner, metric='euclidean')


fitDist = cdist(fitnessTest, fitnessTest, metric='euclidean')


# import pandas as pd
# pd.DataFrame(dists).to_csv('dists_'+model_name+representation)
# df = pd.read_csv("outputfilename"+representation,sep='\t',skiprows=(0,1,2,3),header=None).drop(columns=[0,1]).values
# df2 = pd.read_csv("outputfilename"+representation+"-2",sep='\t',skiprows=(0,1,2,3),header=None).drop(columns=[0,1]).values

inds = np.triu_indices(howMany, 1)

valsDists = dists[inds]
# valsFit = df[inds]
# valsFit2 = df2[inds]
# valsFit3 = valsFit2 - valsFit
# valsRand = np.random.rand(np.shape(valsFit)[0])/2.5 - 0.15

valsFitness = fitDist[inds]

# corr = np.corrcoef(valsDists, valsFit)
# corr2 = np.corrcoef(valsDists, valsFit2)
# corr3 = np.corrcoef(valsDists, valsFit3)

corrFitness = np.corrcoef(valsDists, valsFitness)
# corr1 = np.corrcoef(valsDists, valsRand)
# corr2 = np.corrcoef(valsRand, valsFit)
print("Correlation:")
# print(corr)
# print(corr2)
# print(corr3)
print(corrFitness)

# for i in range(len(genos)):
#     gen2[i] = np.delete(gen2[i], np.where(zerosTest[i]==0))
#     genos[i] = np.delete(genos[i], np.where(zerosTest[i]==0))
# print("genos now")
# for i in range(200):
#
#     print("geno " + str(i))
#     print("-----original-------")
#     print(np.delete(gen2Check[i], np.where(zerosTestCheck[i]==0)))
#     print("-----predicted------")
#     #print(genosCheck[i])
#     print(np.delete(genosCheck[i], np.where(zerosTestCheck[i]==0)))
#     print("-----encoded--------")
#     print([outH[i], outC[i]])
#     print("-----predicted FULL------")
#     # print(genosCheck[i])
#     print(np.delete(genosCheck[i], np.where(zerosTestReadCheck[i] == 0)))

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





    dec = decoder.predict([np.expand_dims(hiddenH, axis=0), np.expand_dims(hiddenC, axis=0), np.ones(shape=(1,max_len,32))])
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

