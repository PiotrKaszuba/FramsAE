# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l1_l2
import numpy as np
from tensorflow.keras.initializers import he_normal, he_uniform

def createModel(max_len, features, dimsEmbed, lr, two_layer=False, bidir=False, cells=32, regularization_base=2e-6):
    inp = Input(shape=(max_len,))
    inp2 = Input(shape=(max_len,))

    embedLayer = Embedding(features + 1, dimsEmbed, input_length=max_len, embeddings_initializer=he_uniform(2), mask_zero=True, name='inpEmbed1')
    prev_layer = embedLayer(inp)

    embed2Layer = Embedding(2, cells, input_length=max_len, embeddings_initializer=he_uniform(3), mask_zero=True, trainable=False, name='inpEmbed2')
    embed2 = embed2Layer(inp2)

    embed2Layer.set_weights(np.expand_dims(
        np.stack([np.zeros_like(embed2Layer.get_weights()[0][0]), np.ones_like(embed2Layer.get_weights()[0][0])],
                 axis=0), axis=0))

    if two_layer:
        lstmEncSecLayer = LSTM(cells, activation='relu', name='encoder2Layer', return_sequences=True,
                               kernel_initializer=he_normal(1),
                               kernel_regularizer=l1_l2(regularization_base, regularization_base),
                               bias_regularizer=l1_l2(2 * regularization_base, 2 * regularization_base))

        if bidir:
            lstmEncSecLayer = Bidirectional(lstmEncSecLayer)

        prev_layer = lstmEncSecLayer(prev_layer)

    lstmEncLayer = LSTM(cells, activation='relu', return_sequences=False, kernel_initializer=he_normal(5),
                        name='encoderLayer',
                        return_state=True,
                        recurrent_regularizer=l1_l2(regularization_base / 20, regularization_base / 20),
                        kernel_regularizer=l1_l2(regularization_base, regularization_base),
                        bias_regularizer=l1_l2(regularization_base * 2, regularization_base * 2))

    if bidir:
        lstmEncLayer = Bidirectional(lstmEncLayer)
        enc, h1, h2, h3, h4 = lstmEncLayer(prev_layer)
        concat = Concatenate()([h1, h2, h3, h4])

    else:
        enc, h1, c1 = lstmEncLayer(prev_layer)
        concat = Concatenate()([h1, c1])

    bn1 = BatchNormalization(name='bn1')(concat)

    hLayer = Dense(cells, activation='tanh', use_bias=True, kernel_initializer=he_normal(10), name='hDense',
                   activity_regularizer=l1_l2(regularization_base / 2, regularization_base / 2),
                   bias_regularizer=l1_l2(regularization_base * 2.5, regularization_base * 2.5))
    h = hLayer(bn1)

    cLayer = Dense(cells, activation='linear', use_bias=True, kernel_initializer=he_normal(11), name='cDense',
                   activity_regularizer=l1_l2(regularization_base / 2, regularization_base / 2),
                   bias_regularizer=l1_l2(regularization_base * 2.5, regularization_base * 2.5))
    c = cLayer(bn1)

    decoderInpH = Input((cells,))
    decoderInpC = Input((cells,))
    decoderPrevInput = Input(((max_len, cells)))

    rep = RepeatVector(max_len)(decoderInpH)
    mult = Multiply()([decoderPrevInput, rep])
    mask = Masking(0.0)(mult)

    prev_layer = LSTM(cells, activation='relu', return_sequences=True, kernel_initializer=he_normal(21), name='decoder1',
                      kernel_regularizer=l1_l2(regularization_base * 1.5, regularization_base * 1.5),
                      bias_regularizer=l1_l2(regularization_base * 2.5, regularization_base * 2.5))(mask,
                                                                                                    initial_state=[
                                                                                                        decoderInpH,
                                                                                                        decoderInpC])
    if two_layer:
        prev_layer = LSTM(cells, activation='relu', return_sequences=True, kernel_initializer=he_normal(35),
                          name='decoder2',
                          kernel_regularizer=l1_l2(regularization_base * 1.5, regularization_base * 1.5),
                          bias_regularizer=l1_l2(regularization_base * 2.5, regularization_base * 2.5))(prev_layer)

    out = TimeDistributed(Dense(features, activation='softmax', kernel_initializer=he_normal(100), name='denseOut'),
                          name='denseOut')(prev_layer)

    decoder = Model(inputs=[decoderInpH, decoderInpC, decoderPrevInput], outputs=out)
    model = Model(inputs=[inp, inp2], outputs=decoder([h, c, embed2]))

    model.summary()
    decoder.summary()
    model.compile(optimizer=Adam(lr, clipnorm=0.5), loss='categorical_crossentropy')
    decoder.compile(optimizer=Adam(lr, clipnorm=0.5), loss='categorical_crossentropy')

    encoder = Model(inputs=[inp, inp2], outputs=[h, c, embed2])

    return model, encoder, decoder


def load_weights(model, decoder, weights_path):
    model.load_weights(weights_path)
    #decoder.load_weights(weights_path)

    print("Loaded weights from: " + str(weights_path))


    # if past_epochs > 0:
        # if bidir and twoLstmLayers:
        #     if representation == 'f1':
        #         model.load_weights('models/modelBiDir22509_0.9718959477721824_weights_spare', by_name=True)
        #         decoder.load_weights('models/modelBiDir22509_0.9718959477721824_weights_spare', by_name=True)
        #     if representation == 'f4':
        #         pass
        #         model.load_weights('models/modelBiDirf4_3780_0.7178693819185917_weights', by_name=True)
        #         decoder.load_weights('models/modelBiDirf4_3780_0.7178693819185917_weights', by_name=True)
        #     if representation == 'f9':
        #         model.load_weights('models/modelBiDirf9_8000_0.9873158029708243_weights', by_name=True)
        #         decoder.load_weights('models/modelBiDirf9_8000_0.9873158029708243_weights', by_name=True)
        #
        # if not bidir and not twoLstmLayers:
        #     if representation == 'f1':
        #         pass
        #         print("loaded")
        #         model.load_weights('models/modelOneLayerf1_12700_0.8167521666341699_weights', by_name=True)
        #         decoder.load_weights('models/modelOneLayerf1_12700_0.8167521666341699_weights', by_name=True)
        #         # model.load_weights('models/modelOneLayerf1_27300_0.8691816879912326_weights', by_name=True)
        #         # decoder.load_weights('models/modelOneLayerf1_27300_0.8691816879912326_weights', by_name=True)
        #
        # if bidir and not twoLstmLayers:
        #     if representation == 'f1':
        #         pass
        #
        #         model.load_weights('models/modelOneLayerBidirf1_9000_0.8579871313024009_weights', by_name=True)
        #         decoder.load_weights('models/modelOneLayerBidirf1_9000_0.8579871313024009_weights', by_name=True)
        #         # model.load_weights('models/' + model_name + representation + '_' + str(past_epochs) + '_' + str(loaded_model_acc) + '_weights',by_name=True)
        #         # decoder.load_weights('models/'+model_name+representation+'_'+str(past_epochs) + '_' + str(loaded_model_acc) + '_weights', by_name=True)
