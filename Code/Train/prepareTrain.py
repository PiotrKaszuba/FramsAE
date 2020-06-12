from Code.Preparation.ParseGenFile import testGenos
from Code.Preparation.encodeGenos import prepareEncoders,encodeGenos, inverse
import tensorflow as tf
pad_sequences = tf.keras.preprocessing.sequence.pad_sequences
import os, glob
import numpy as np
import pickle
from Code.Preparation.createModel import createModel, load_weights
from Code.Statistics.makeHistograms import plot, plot_acc_by_length
import random
def RepresentsInt(s):
    try:
        int(s)
        return True
    except ValueError:
        return False

def prepareTrain(config):
    os.environ['PYTHONHASHSEED'] = '0'
    np.random.seed(42)
    random.seed(12345)
    tf.random.set_seed(1)
    reverse = config['reverse']
    # load genos
    genos = testGenos(config, print_some_genos=True)

    temp = config['long_genos']
    config['long_genos'] = 'TEST'
    genosTest = testGenos(config, print_some_genos=True)
    config['long_genos'] = temp

    # get symbol encoders from dictionary
    encoders = prepareEncoders(config['dict'])

    # prepare in / out sequences
    sequences = encodeGenos(genos, encoders, config['oneHot'])
    sequences2 = encodeGenos(genos, encoders, True)

    sequencesTest = encodeGenos(genosTest, encoders, config['oneHot'])
    sequences2Test = encodeGenos(genosTest, encoders, True)


    # preparing inputs, with neutral value on zero and then moving it to 0 (necessary for decoding later)


    X = pad_sequences(sequences, padding='pre', dtype='int32', value=-1)
    X+=1

    XTest = pad_sequences(sequencesTest, padding='pre', dtype='int32', value=-1)
    XTest += 1

    # preapring outputs
    if reverse:
        Y = pad_sequences(sequences2, padding='pre', dtype='float32', value=0.0)
        Y = np.flip(Y, 1) # (inverse of inputs)

        YTest = pad_sequences(sequences2Test, padding='pre', dtype='float32', value=0.0)
        YTest = np.flip(YTest, 1)  # (inverse of inputs)
    else:
        Y = pad_sequences(sequences2, padding='post', dtype='float32', value=0.0)
        YTest = pad_sequences(sequences2Test, padding='post', dtype='float32', value=0.0)


    # inputs for masking purposes
    zerosInputs = np.where(X == 0, 0, 1)
    zerosInputs = np.flip(zerosInputs, 1)

    zerosInputsTest = np.where(XTest == 0, 0, 1)
    zerosInputsTest = np.flip(zerosInputsTest, 1)

    #X_train, X_test, y_train, y_test, zerosTrain, zerosTest, genosTrain, genosTest = train_test_split(X, Y,zerosInputs, genos, test_size=0.2, random_state=42)
    genosTrain = genos
    genosTest = genosTest

    X_train = X
    X_test = XTest

    y_train = Y
    y_test = YTest

    zerosTrain = zerosInputs
    zerosTest = zerosInputsTest

    max_len = np.shape(X)[1]

    if reverse:
        gen2Check=pad_sequences(np.array([list(gen[::-1]) for gen in genosTest]), padding='post', dtype=object, value='', maxlen=np.shape(X)[1])
    else:
        gen2Check=pad_sequences(np.array([list(gen) for gen in genosTest]), padding='post', dtype=object, value='', maxlen=np.shape(X)[1])

    # if reverse:
    #     X_testCheck = X_test - 1
    #
    #     X_testCheck = np.where(X_testCheck == -1, 0, X_testCheck)
    #     gen2Check = inverse(X_testCheck, encoders, False)
    #     gen2Check = np.array(gen2Check)

    # zerosTestCheck = np.flip(zerosTest, 1)
    zerosTestCheck = zerosTest

    model, encoder, decoder = createModel(max_len, config['features'], config['dimsEmbed'], config['lr'], config['twoLayer'], config['bidir'], config['cells'], config['reg_base'])
    #model_path_base = os.path.join(config['data_path'], config['model_name'] + str(config['past_epochs'])) + "_" + str(config['loaded_model_acc'])

    model_path_name = os.path.join(config['data_path'], config['model_name'])


    losses_path = None
    expected_epochs = None


    if os.path.exists(model_path_name + '_losses' + '_tmp'):
        losses_path = model_path_name + '_losses' + '_tmp'
    else:
        if os.path.exists(model_path_name + '_losses'):
            losses_path = model_path_name + '_losses'


    if losses_path is not None:
        with open(losses_path,'rb') as handle:
            losses_file = pickle.load(handle)

        hist = losses_file['history']
        accs = losses_file['accuracy']
        eps_tot = losses_file['epochs_total']
        expected_epochs = eps_tot[-1]
        print("Loaded losses at epochs: " + str(expected_epochs))
        if config['onlyEval'] == 'True':
            plot(hist, accs, eps_tot, config['model_name'], config['load_dir'])
    else:
        hist = []
        accs = []
        eps_tot = []



    files = []
    for filename in glob.glob(model_path_name + "*"):
            files.append('.'.join(filename.split('.')[:-1]))

    files = list(set(files))

    if len(files)>0:
        files = sorted([(f,int(f[len(model_path_name):].split('_')[0])) for f in files if RepresentsInt(f[len(model_path_name):].split('_')[0])], key=lambda x: x[1])

        break_ind = -1
        if expected_epochs is not None:
            for ind,f in enumerate(files):
                if f[1] == expected_epochs:
                    break_ind = ind

        files = [f for f, _ in files]

        savedFiles = files[:break_ind+1]

        # for fDel in [f for f in files if f not in savedFiles]:
        #     for filename in glob.glob(fDel+"*"):
        #         print("Deleting not matching weights file: " + str(filename))
        #         os.remove(filename)

        if len(savedFiles)>0:
            print("Loaded saved files: " + str(savedFiles))
            load_weights(model, decoder, savedFiles[-1])
        else:
            print("Did not find matching save file to last loss!")
            hist = []
            accs = []
            eps_tot = []
    else:
        savedFiles = []


    if expected_epochs is not None:
        config['past_epochs'] = expected_epochs



    # if config['past_epochs'] > 0:
    #     load_weights(model, decoder, model_path_base)


    if config['onlyEval'] == 'True':
        evaluate(model, [X_test, zerosTest], gen2Check, zerosTestCheck, encoders, reverse, prepare_hist=True,
                 config=config)
        return
    evaluate(model, [X_test, zerosTest], gen2Check, zerosTestCheck, encoders, reverse)


    # savedAcc = 0.26 # REMOVE
    for i in range(config['epochs']):
        history = model.fit([X_train, zerosTrain], y_train,
                            epochs=config['epochs_per_i'],
                            verbose=2,
                            validation_data=([X_test, zerosTest], y_test),
                            shuffle=True, batch_size=config['batch_size'])

        _, acc = evaluate(model, [X_test, zerosTest], gen2Check, zerosTestCheck, encoders, reverse)

        epochs_total = (i + 1) * config['epochs_per_i'] + config['past_epochs']
        print("Total epochs done: " + str(epochs_total))

        model_path_base = model_path_name + str(epochs_total)

        #model.save( model_path_base + '_' + str(acc) ,True, True)


        weights = model_path_base + '_' + str(acc) + '_weights'
        model.save_weights(weights)

        # if acc < savedAcc - 0.1:  # REMOVE
        #     model.save_weights(os.path.join(config['data_path'],'CHECK'+str(epochs_total)))
        # if acc> savedAcc:
        #     savedAcc = acc

        accs.append(acc)
        hist.append(history.history)
        eps_tot.append(epochs_total)
        d = {'accuracy': accs, 'history': hist, 'epochs_total':eps_tot}

        losses = model_path_name+'_losses'

        with open(losses+'_tmp','wb') as handle:
            pickle.dump(d, handle)

        if os.path.exists(losses):
            os.remove(losses)

        os.rename(losses+'_tmp', losses)

        savedFiles.append(weights)
        if len(savedFiles)> 3:
            for filename in glob.glob(savedFiles[0]+"*"):
                os.remove(filename)
            del savedFiles[0]







def evaluate(model, testData, gen2Check, zerosTestCheck, encoders, reverse=False, prepare_hist=False, config=None):
    res = model.predict(testData, batch_size=2048)
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

    # if reverse:
    #     genosCheck = np.flip(genosCheck, 1)


    genosCheck = np.array(genosCheck, dtype=object)

    sumOfTokens = np.sum(zerosTestCheck)

    hits = np.where((zerosTestCheck == 1) & (gen2Check == genosCheck), 1, 0)
    sumOfHits = hits.sum()

    acc = sumOfHits / sumOfTokens

    if prepare_hist and config is not None:
        plot_acc_by_length(acc, hits, zerosTestCheck, model_name=config['model_name'], path=config['load_dir'])


    print("Token accuracy: " + str(acc))

    return genosCheck, acc


