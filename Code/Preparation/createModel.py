# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l1_l2
import numpy as np
from tensorflow.keras.initializers import he_normal, he_uniform
import tensorflow as tf


# from tensorflow.keras.losses import categorical_crossentropy
# def custom_loss(y_actual,y_pred):
#     custom_loss=categorical_crossentropy(y_actual, y_pred)
#     return custom_loss
#


from Code.Preparation.Utils import locality1_op, locality2_op, locality_term_op


# def get_gradient_norm(model):
#     with K.name_scope('gradient_norm'):
#         grads = K.gradients(model.total_loss, model.trainable_weights)
#         norm = K.sqrt(sum([K.sum(K.square(g)) for g in grads]))
#     return norm

def fitnessLearnModel(max_len, features, lr, cells=32, regularization_base=2e-6):
    inp = Input(shape=(max_len, features), name='fitnessModel_inputs1')
    inp2 = Input(shape=(max_len, features), name='fitnessModel_inputs2')

    mult = Multiply()([inp, inp2])

    mask = Masking(0.0)(mult)
    lstm_Layer = LSTM(cells, activation='relu', return_sequences=False, kernel_initializer=he_normal(24353),
                        name='fitnessModel_lstm1',
                        return_state=False,
                        recurrent_regularizer=l1_l2(regularization_base / 20, regularization_base / 20),
                        kernel_regularizer=l1_l2(regularization_base, regularization_base),
                        bias_regularizer=l1_l2(regularization_base * 2, regularization_base * 2))

    lstm_out = lstm_Layer(mask)
    out = Dense(1, activation='linear', kernel_initializer=he_normal(53436), name='fitnessModel_denseOut')(lstm_out)

    fitnessModel = Model(inputs=[inp, inp2], outputs=out)
    fitnessModel.compile(optimizer=Adam(lr, clipnorm=1.0, clipvalue=0.5), loss='mse')
    return fitnessModel

def createModel(max_len, features, dimsEmbed, lr, two_layer=False, bidir=False, cells=32, regularization_base=2e-6, locality_term = False, batch_size = None, locality_power=1, **kwargs):
    print(tf.executing_eagerly())
    inp = Input(shape=(max_len,), name="inputs1")
    inp2 = Input(shape=(max_len,), name="inputs2")

    inp3 = Input(shape=(batch_size,batch_size), name="inputs3")

    if 'regularization_base_latent' in kwargs:
        regularization_base_latent = kwargs['regularization_base_latent']
    else:
        regularization_base_latent = regularization_base

    embedLayer = Embedding(features+1, dimsEmbed, input_length=max_len, embeddings_initializer=he_uniform(2), mask_zero=True, name='inpEmbed1')
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
                   activity_regularizer=l1_l2(regularization_base_latent / 2, regularization_base_latent / 2),
                   bias_regularizer=l1_l2(regularization_base_latent * 2.5, regularization_base_latent * 2.5))
    h = hLayer(bn1)

    cLayer = Dense(cells, activation='linear', use_bias=True, kernel_initializer=he_normal(11), name='cDense',
                   activity_regularizer=l1_l2(regularization_base_latent / 2, regularization_base_latent / 2),
                   bias_regularizer=l1_l2(regularization_base_latent * 2.5, regularization_base_latent * 2.5))
    c = cLayer(bn1)

    if locality_term:
        locality1 = Lambda(locality1_op)([h, c])
        locality2 = Lambda(locality2_op)(inp3)
        locality_layer = Lambda(locality_term_op)([locality1, locality2])

    decoderInpH = Input((cells,))
    decoderInpC = Input((cells,))
    decoderPrevInput = Input(((max_len, cells)))
    print("model kwargs: ", kwargs)
    timeWindowsConstant = False if 'timeWindowsConstant' not in kwargs else kwargs['timeWindowsConstant']
    decoderInpDenses = False if 'decoderInpDenses' not in kwargs else kwargs['decoderInpDenses']
    inpHZeros = False if 'inpHZeros' not in kwargs else kwargs['inpHZeros']
    inpCZeros = False if 'inpCZeros' not in kwargs else kwargs['inpCZeros']
    outAdditionalDense = False if 'outAdditionalDense' not in kwargs else kwargs['outAdditionalDense']

    decoderInpH_topass = decoderInpH
    decoderInpC_topass = decoderInpC

    if timeWindowsConstant:
        rep = decoderPrevInput
    else:
        rep = RepeatVector(max_len)(decoderInpH)
    # mult = Multiply()([decoderPrevInput, decoderPrevInput])

    if decoderInpDenses:

        decoderInpHDense1 = Dense(cells, activation='relu', kernel_initializer=he_normal(32678), name='decInpHDense1',
                                  kernel_regularizer=l1_l2(regularization_base*1.0, regularization_base*1.0),
                                  bias_regularizer=l1_l2(regularization_base*1.0, regularization_base*1.0))(decoderInpH)

        decoderInpHDense2 = Dense(cells, activation='relu', kernel_initializer=he_normal(32679), name='decInpHDense2',
                                  kernel_regularizer=l1_l2(regularization_base * 1.0, regularization_base * 1.0),
                                  bias_regularizer=l1_l2(regularization_base * 1.0, regularization_base * 1.0))(decoderInpHDense1)

        decoderInpH_topass = Dense(cells, activation='tanh', kernel_initializer=he_normal(72679), name='decInpHDense3',
                                  kernel_regularizer=l1_l2(regularization_base * 1.0, regularization_base * 1.0),
                                  bias_regularizer=l1_l2(regularization_base * 1.0, regularization_base * 1.0))(
            decoderInpHDense2)


        decoderInpCDense1 = Dense(cells, activation='relu', kernel_initializer=he_normal(32618), name='decInpCDense1',
                                  kernel_regularizer=l1_l2(regularization_base * 1.0, regularization_base * 1.0),
                                  bias_regularizer=l1_l2(regularization_base * 1.0, regularization_base * 1.0))(decoderInpC)

        decoderInpCDense2 = Dense(cells, activation='relu', kernel_initializer=he_normal(32628), name='decInpCDense2',
                                  kernel_regularizer=l1_l2(regularization_base * 1.0, regularization_base * 1.0),
                                  bias_regularizer=l1_l2(regularization_base * 1.0, regularization_base * 1.0))(decoderInpCDense1)

        decoderInpC_topass = Dense(cells, activation='linear', kernel_initializer=he_normal(32619), name='decInpCDense3',
                                  kernel_regularizer=l1_l2(regularization_base * 1.0, regularization_base * 1.0),
                                  bias_regularizer=l1_l2(regularization_base * 1.0, regularization_base * 1.0))(
            decoderInpCDense2)

    if inpHZeros:
        decoderInpH_topass = Lambda(lambda x: x*0.0)(decoderInpH_topass)
    if inpCZeros:
        decoderInpC_topass = Lambda(lambda x: x * 0.0)(decoderInpC_topass)
    # if not timeWindowsConstant:

    mult = Multiply()([rep, decoderPrevInput])

    mask = Masking(0.0)(mult)

    prev_layer = LSTM(cells, activation='relu', return_sequences=True, kernel_initializer=he_normal(21), name='decoder1',
                      kernel_regularizer=l1_l2(regularization_base * 1.5, regularization_base * 1.5),
                      bias_regularizer=l1_l2(regularization_base * 2.5, regularization_base * 2.5))(mask,
                                                                                                    initial_state=[
                                                                                                        decoderInpH_topass,
                                                                                                        decoderInpC_topass])
    if two_layer:
        prev_layer = LSTM(cells, activation='relu', return_sequences=True, kernel_initializer=he_normal(35),
                          name='decoder2',
                          kernel_regularizer=l1_l2(regularization_base * 1.5, regularization_base * 1.5),
                          bias_regularizer=l1_l2(regularization_base * 2.5, regularization_base * 2.5))(prev_layer)
    # to comment

    if outAdditionalDense:

        outPrev = TimeDistributed(Dense(32, activation='relu', kernel_initializer=he_normal(836), name='densePrevOut',
                                        kernel_regularizer=l1_l2(regularization_base * 2.0, regularization_base * 2.0),
                                        bias_regularizer=l1_l2(regularization_base * 2.5, regularization_base * 2.5)
                                        ), name='densePrevOut')(prev_layer)
        out = TimeDistributed(Dense(features, activation='softmax', kernel_initializer=he_normal(100), name='denseOut'),
                              name='denseOut')(outPrev)
    else:
        out = TimeDistributed(Dense(features, activation='softmax', kernel_initializer=he_normal(100), name='denseOut'),
                              name='denseOut')(prev_layer)


    decoder = Model(inputs=[decoderInpH, decoderInpC, decoderPrevInput], outputs=out)
    if locality_term:
        model = Model(inputs=[inp, inp2, inp3], outputs=decoder([h, c, embed2]))
    else:
        model = Model(inputs=[inp, inp2], outputs=decoder([h, c, embed2]))

    model.summary()
    decoder.summary()

    if locality_term:
        print("Using locality term! Locality power: " , locality_power)
        locality_loss = (1-locality_layer)*tf.constant(locality_power)
        model.add_loss(locality_loss)


        model.add_metric(locality_loss, name='locality', aggregation='mean')
    # model.add_metric(get_gradient_norm(model), name='locality', aggregation='mean')
    # model.add_metric(locality_loss, name='localityS', aggregation='sum')
    model.compile(optimizer=Adam(lr, clipnorm=1.0, clipvalue=0.5), loss='categorical_crossentropy')
    decoder.compile(optimizer=Adam(lr, clipnorm=1.0, clipvalue=0.5), loss='categorical_crossentropy')
    # model.metrics_tensors = []
    if locality_term:
        encoder = Model(inputs=[inp, inp2, inp3], outputs=[h, c, embed2])
    else:
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
