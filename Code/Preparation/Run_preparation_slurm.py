import os
import glob
import pickle
import re
from typing import List
import numpy as np
import tensorflow as tf

pad_sequences = tf.keras.preprocessing.sequence.pad_sequences

from ParseGenFile import testGenos
from Utils import get_genSuff, RepresentsInt
from createModel import createModel, load_weights
from encodeGenos import encodeGenos, inverse


class EvolutionModel:

    def __init__(self, config):


        self.genSuff = get_genSuff(config['representation'])

        self.reverse = config['reverse']
        self.model_path_name = os.path.join(config['data_path'], config['model_name'])
        self.config = config


        model, encoder, decoder = createModel(config['max_len'], config['features'], config['dimsEmbed'], config['lr'],
                                              config['twoLayer'], config['bidir'], config['cells'], config['reg_base'])

        self.model = model
        self.encoder = encoder
        self.decoder = decoder

        self.hist = None
        self.accs = None
        self.eps_tot = None
        self.savedFiles = None

        self.load_model_data()


        self.means = None
        self.means_mut = None
        self.cov = None
        self.latent = None
        self.data = None

        self.prepareLatentProperties()

        self.simplest = None


    def evaluate(self, X, genos_true):
        res = self.model.predict(X, batch_size=2048)

        genosCheck = self.get_genos(res)

        sumOfTokens = np.sum(X[1])

        hits = np.where((X[1] == 1) & (genos_true == genosCheck), 1, 0)
        sumOfHits = hits.sum()

        acc = sumOfHits / sumOfTokens

        print("Token accuracy: " + str(acc))

        return acc

    def get_genos(self, dec_predict):
        # get sequences with ones in maxes
        argmax = np.argmax(dec_predict, axis=-1)
        ind = np.indices(argmax.shape)
        argmax = np.expand_dims(argmax, axis=0)
        ind = np.concatenate([ind, argmax], axis=0)
        row, col, argm = ind
        resMax = np.zeros_like(dec_predict)
        resMax[row, col, argm] = 1

        # get genotype for predicted
        genosCheck = inverse(resMax, self.config['encoders'], True)

        genosCheck = np.array(genosCheck)

        return genosCheck

    def predictEncoderLatent(self, X, ret_Full=False):
        enc_predict = self.encoder.predict(X)
        latent = getConcatEncoderPredGenotypes(enc_predict)
        if ret_Full:
            return latent, enc_predict[2], enc_predict[0], enc_predict[1]
        return latent

    def mutate(self, latent, times=1, magnitude=0.01):
        latent = np.copy(latent)
        for i in range(times):
            mutation = np.random.multivariate_normal(self.means_mut, self.cov * magnitude, size=len(latent))
            latent += mutation
        return latent

    def getOnesMaskForFullPrediction(self, samples):
        return np.ones((samples, self.config['max_len'], self.config['cells']))


    def sampleMultivariateGaussianLatent(self, samples):
        sampled = np.random.multivariate_normal(self.means, self.cov, size=samples)
        print(np.max(sampled, axis=0))
        print(np.min(sampled, axis=0))

        return sampled

    def prepareLatentProperties(self):
        if self.means_mut is None:
            self.means_mut = np.zeros((self.config['cells']*2,))

        cov_path = self.model_path_name+"_cov"
        means_path = self.model_path_name+"_means"
        latent_path =  self.model_path_name+"_latent"
        if os.path.exists(cov_path):
            with open(cov_path, 'rb') as handle:
                self.cov = pickle.load(handle)
                print("Loaded cov")
        if os.path.exists(means_path):
            with open(means_path, 'rb') as handle:
                self.means = pickle.load(handle)
                print("Loaded means")

        if os.path.exists(latent_path):
            with open(latent_path, 'rb') as handle:
                self.latent = pickle.load(handle)
                print("Loaded latent")

        self.data = prepareData(self.config)
        if self.cov is None or self.means is None or self.latent is None:
            self.latent = self.predictEncoderLatent(self.data[0])
            self.means = np.mean(self.latent, axis=0)
            self.cov = np.cov(self.latent.T)

            with open(cov_path, 'wb') as handle:
                pickle.dump(self.cov, handle)
            with open(means_path, 'wb') as handle:
                pickle.dump(self.means, handle)
            with open(latent_path, 'wb') as handle:
                pickle.dump(self.latent, handle)


    def load_model_data(self):
        losses_path = None
        expected_epochs = None
        print(self.model_path_name)
        if os.path.exists(self.model_path_name + '_losses' + '_tmp'):
            losses_path = self.model_path_name + '_losses' + '_tmp'
        else:
            if os.path.exists(self.model_path_name + '_losses'):
                losses_path = self.model_path_name + '_losses'

        if losses_path is not None:
            with open(losses_path, 'rb') as handle:
                losses_file = pickle.load(handle)

            self.hist = losses_file['history']
            self.accs = losses_file['accuracy']
            self.eps_tot = losses_file['epochs_total']

            if 'cov' in losses_file:
                self.cov = losses_file['cov']
            if 'means' in losses_file:
                self.means = losses_file['means']

            expected_epochs = self.eps_tot[-1]
            print("Loaded losses at epochs: " + str(expected_epochs))
            # if config['onlyEval'] == 'True':
            #     plot(hist, accs, eps_tot, config['model_name'], config['load_dir'])
        else:
            self.hist = []
            self.accs = []
            self.eps_tot = []

        files = []
        for filename in glob.glob(self.model_path_name + "*"):
            files.append('.'.join(filename.split('.')[:-1]))

        files = list(set(files))

        if len(files) > 0:
            files = sorted([(f, int(f[len(self.model_path_name):].split('_')[0])) for f in files if
                            RepresentsInt(f[len(self.model_path_name):].split('_')[0])], key=lambda x: x[1])

            break_ind = -1
            if expected_epochs is not None:
                for ind, f in enumerate(files):
                    if f[1] == expected_epochs:
                        break_ind = ind

            files = [f for f, _ in files]

            self.savedFiles = files[:break_ind + 1]

            # for fDel in [f for f in files if f not in savedFiles]:
            #     for filename in glob.glob(fDel+"*"):
            #         print("Deleting not matching weights file: " + str(filename))
            #         os.remove(filename)

            if len(self.savedFiles) > 0:
                print("Loaded saved files: " + str(self.savedFiles))
                load_weights(self.model, self.decoder, self.savedFiles[-1])
            else:
                print("Did not find matching save file to last loss!")
                self.hist = []
                self.accs = []
                self.eps_tot = []
        else:
            self.savedFiles = []

        if expected_epochs is not None:
            self.config['past_epochs'] = expected_epochs


#####################

# assume batch axis is there
def splitLatentToHiddenAndCell(latent):
    ln = int(np.shape(latent)[1]/2)
    hidden = latent[:, 0:ln]
    cell = latent[:, ln:]
    return hidden, cell
def getConcatEncoderPredGenotypes(latent):
    return np.concatenate([latent[0], latent[1]], axis=1)

def extractIthGenotypeFromEncoderPred(latent, i):
    return getConcatEncoderPredGenotypes(latent)[i]

def prepareGenos(genos : List[str], config):
    sequences = encodeGenos(genos, config['encoders'], config['oneHot'])
    sequences2 = encodeGenos(genos, config['encoders'], True)
    X = pad_sequences(sequences, padding='pre', dtype='int32', value=-1, maxlen=config['max_len'])
    X += 1

    if config['reverse']:
        Y = pad_sequences(sequences2, padding='pre', dtype='float32', value=0.0, maxlen=config['max_len'])
        Y = np.flip(Y, 1)  # (inverse of inputs)
    else:
        Y = pad_sequences(sequences2, padding='post', dtype='float32', value=0.0, maxlen=config['max_len'])

    zerosInputs = np.where(X == 0, 0, 1)
    zerosInputs = np.flip(zerosInputs, 1)

    if config['reverse']:
        genosCheck = pad_sequences(np.array([list(gen[::-1]) for gen in genos]), padding='post', dtype=object,
                                   value='', maxlen=config['max_len'])
    else:
        genosCheck = pad_sequences(np.array([list(gen) for gen in genos]), padding='post', dtype=object, value='',
                                   maxlen=config['max_len'])

    return [X, zerosInputs], Y, genosCheck

def prepareData(config):
    genos = testGenos(config, print_some_genos=True)

    return prepareGenos(genos, config)
