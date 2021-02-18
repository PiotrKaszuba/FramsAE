import os
import glob
import pickle
from typing import List
import numpy as np

import shutil
from collections import defaultdict, deque, Counter


from Code.Preparation.Utils import get_genSuff, RepresentsInt, extract_fram, prepareData, extractFitnesses, prepareGenos
from Code.Preparation.createModel import createModel, load_weights, fitnessLearnModel
from Code.Preparation.encodeGenos import inverse
from Code.FramsticksCli import FramsticksCLI, fake_fitness
from scipy import stats
class EvolutionModel:

    def __init__(self, config, load_autoencoder=True):


        self.genSuff = get_genSuff(config['representation'])

        self.config = config
        self.fitnessDict = {}
        self.geno_bank = defaultdict(list)
        self.starting_genos = deque(maxlen=config['starting_points'])
        self.genos = None
        self.simplest = None
        self.framsCLI: FramsticksCLI = None
        self.cli_orchestra : CLIOrchestra = CLIOrchestra(self.config)
        self.validCnt = Counter()
        self.invalidCnt = Counter()
        if self.config['representation'] == 'f1':
            self.fill_geno_bank()
            self.fill_geno_bank()
        dataProvider = config['dataProvider'](config['train_data_get_methods_maker'],
                                              config['test_data_get_methods_maker'], config, em=self)
        _, ask_for_objects_f = dataProvider.get_test_data()

        self.data = (ask_for_objects_f('input'), ask_for_objects_f('output'), ask_for_objects_f('genosCheck'))
        self.genos = ask_for_objects_f('genos')
        # try:
        #     self.data, self.genos = prepareData(self.config, ret_pure_genos=True)
        # except:
        #     dataProvider = config['dataProvider'](config['train_data_get_methods_maker'],
        #                                                            config['test_data_get_methods_maker'], config, em=self)
        #     _, ask_for_objects_f = dataProvider.get_test_data()
        #
        #
        #     self.data = (ask_for_objects_f('input'), ask_for_objects_f('output'), ask_for_objects_f('genosCheck'))
        #     self.genos = ask_for_objects_f('genos')


        if load_autoencoder:
            self.reverse = config['reverse']
            self.model_path_name = os.path.join(config['data_path'], config['model_name'])
            self.fitnessModel_path_name = os.path.join(config['data_path'], config['representation']+'_'+str(fake_fitness[0]))
            model, encoder, decoder = createModel(config['max_len'], config['features'], config['dimsEmbed'], config['lr'],
                                              config['twoLayer'], config['bidir'], config['cells'], config['reg_base'],
                                              config['locality_term'], config['batch_size'], config['locality_power'], **config['model_kwargs'])
            # self.fitnessModel = fitnessLearnModel(config['max_len'], config['features'], config['lr'], config['cells'], config['reg_base'])

            self.model = model
            self.encoder = encoder
            self.decoder = decoder

            self.hist = None
            self.accs = None
            self.eps_tot = None
            self.savedFiles = None

            self.histFitn = []
            self.accsFitn = []
            self.eps_totFitn = []
            self.savedFilesFitn = []

            self.load_model_data()


            self.means = None
            self.means_mut = None
            self.cov = None
            self.latent = None
            self.clear_files = config['clear_files']

            self.prepareLatentProperties()
            self.epochs_total = self.config['past_epochs']
            self.epochs_total_fitnessModel = 0
        else:
            self.config['model_name'] = self.config['representation']

    # def valid_check(self, genos: List[str], return_bad_and_inds = False):
    #     if self.framsCLI is None:
    #         self.createCLI()
    #     to_check = [geno for geno in genos if self.validDict.get(geno) is None]
    #     if len(to_check) > 0:
    #         if return_bad_and_inds:
    #             fitnesses, good_inds, bad_mutants = evaluate_with_batches(self, to_check, base_fitness=False, return_bad=return_bad_and_inds)
    #         else:
    #             fitnesses = evaluate_with_batches(self, to_check, base_fitness=False, return_bad=return_bad_and_inds)
    #         fitnesses = [fitn if fitn is not False else None for fitn in fitnesses]
    #         for geno, fitn in zip(to_check, fitnesses):
    #             self.fitnessDict[geno] = fitn
    #     else:
    #         bad_mutants =0
    #         good_inds = list(range(len(genos)))
    #
    #     to_ret= [self.fitnessDict[geno] for geno in genos]
    #     if return_bad_and_inds:
    #         return to_ret, bad_mutants, good_inds
    #     else:
    #         return to_ret

    def fitness_check(self, genos : List[str], return_bad_and_inds = False, raise_err = False):
        if self.framsCLI is None:
            self.createCLI()
        # indices_checking = [i for i,geno in enumerate(genos) if self.fitnessDict.get(geno) is None]
        # self.fitnessDict = {}
        already_good_inds = [ind for ind,geno in enumerate(genos) if self.fitnessDict.get(geno) is not None]
        to_check_inds = [ind for ind,geno in enumerate(genos) if self.fitnessDict.get(geno) is None]
        to_check = [geno for geno in genos if self.fitnessDict.get(geno) is None]
        if len(to_check) > 0:
            if return_bad_and_inds:
                fitnesses, good_inds, bad_mutants = evaluate_with_batches(self, to_check, base_fitness=False, return_bad=return_bad_and_inds, ind_base=np.array(to_check_inds))
                already_good_inds.extend(good_inds)
                good_inds = sorted(already_good_inds)
            else:
                fitnesses = evaluate_with_batches(self, to_check, base_fitness=False, return_bad=return_bad_and_inds)
            fitnesses = [fitn if fitn is not False else None for fitn in fitnesses]
            for geno, fitn in zip(to_check, fitnesses):
                self.fitnessDict[geno] = fitn
        else:
            bad_mutants =0
            good_inds = list(range(len(genos)))

        to_ret= [self.fitnessDict[geno] for geno in genos]
        for i in range(len(genos)):
            if to_ret[i] is None and raise_err:
                raise Exception("ret None")
            if to_ret[i] is None and i in good_inds:
                print('---removed ', i, ' ----')
                good_inds.remove(i)
        if return_bad_and_inds:
            return to_ret, bad_mutants, good_inds
        else:
            return to_ret




    def postProcessFram(self, framNumericList):
        self.config['encoders']

    def prepare_fitnessPrediction(self, inputs, k, include_discete=True):
        decoder_outputs = self.model.predict(inputs, batch_size = self.config['batch_size'])

        frams = []
        for individual in decoder_outputs:
            current_fram = []
            for i in range(k):
                fram = []
                for postition in individual:
                    symbol = np.random.choice(np.arange(self.config['features']), p=postition)
                    fram.append(symbol)
                current_fram.append(fram)
            frams.append(current_fram)


        markers = inputs[1]
        onesMask = np.ones((self.config['features'], markers.shape[0], markers.shape[1]))
        zerosMask = np.zeros((self.config['features'], markers.shape[0], markers.shape[1]))

        mask = np.where(markers, onesMask, zerosMask).transpose([1, 2, 0])

        genos, discrete = self.get_genos(decoder_outputs, return_discrete=include_discete)
        frams = [extract_fram(geno) for geno in genos]

        fitnesses = self.fitness_check(frams)
        inputs = np.concatenate([decoder_outputs, discrete])
        fitnesses = np.concatenate([fitnesses, fitnesses])
        masks = np.concatenate([mask, mask])
        return [inputs, masks], fitnesses

        # for i in range(k):
        #     temp_dec = np.copy(decoder_outputs)
        #     maxes = np.max(temp_dec, axis=-1)
        #
        #     for j in range(i):



    def createCLI(self, level=0):
        if level > 20:
            raise ConnectionError("FramsCLI recursion over 20")
        try:
            if self.framsCLI is not None and level < 10:
                self.framsCLI.closeFramsticksCLI()
            framsCLId = FramsticksCLI(self.config['frams_path'], None if 'framsexe' not in self.config else self.config['framsexe'],
                                      pid=self.config['model_name'] + "_"+ str(level) + (
                                          '' if 'pid_extension' not in self.config else self.config['pid_extension']),
                                      importSim=self.config['importSim'], markers=self.config['markers'], config=self.config)
            print("OK_CORRECTLY CREATED__", level)
        except Exception as e:
            self.createCLI(level=level + 1)
            return
        self.framsCLI = framsCLId
        self.framsCLI.invalidCnt = self.invalidCnt
        self.framsCLI.validCnt = self.validCnt
        self.framsCLI.em = self


    def get_input_from_genos(self, genos : List[str]):
        ([X, zerosInputs], Y, genosCheck) = prepareGenos(genos, self.config)
        if self.config['locality_term']:
            locality_vals = np.zeros((len(genos), self.config['batch_size'], self.config['batch_size']))
            return [X, zerosInputs, locality_vals]
        else:
            return [X,zerosInputs]


    def summarize_train_phase_fitnessModel(self, tested_acc, ):
        self.epochs_total_fitnessModel += self.config['epochs_per_i']
        print("Total epochs done, fitnessModel: " + str(self.epochs_total_fitnessModel))

        model_path_base = self.fitnessModel_path_name + str(self.epochs_total_fitnessModel)

        # model.save( model_path_base + '_' + str(acc) ,True, True)

        weights = model_path_base + '_' + str(tested_acc) + '_weights'
        # weights = model_path_base + '_weights'
        self.fitnessModel.save_weights(weights)

        # if acc < savedAcc - 0.1:  # REMOVE
        #     model.save_weights(os.path.join(config['data_path'],'CHECK'+str(epochs_total)))
        # if acc> savedAcc:
        #     savedAcc = acc

        self.accsFitn.append(tested_acc)
        # self.histFitn.append(history)
        self.eps_totFitn.append(self.epochs_total)
        d = {'accuracy': self.accs, 'history': self.hist, 'epochs_total': self.eps_tot}

        losses = self.fitnessModel_path_name + '_losses'

        with open(losses + '_tmp', 'wb') as handle:
            pickle.dump(d, handle)

        if os.path.exists(losses):
            os.remove(losses)

        os.rename(losses + '_tmp', losses)

        self.savedFilesFitn.append(weights)
        if len(self.savedFilesFitn) > 3:
            for filename in glob.glob(self.savedFilesFitn[0] + "*"):
                try:
                    os.remove(filename)
                except IsADirectoryError:
                    shutil.rmtree(filename)
            del self.savedFilesFitn[0]

    def summarize_train_phase(self, tested_acc, history, i_param):
        self.epochs_total += self.config['epochs_per_i']
        print("Total epochs done: " + str(self.epochs_total))
        self.accs.append(tested_acc)
        self.hist.append(history)
        self.eps_tot.append(self.epochs_total)

        if i_param % self.config['saveFilesEveryITimes'] == 0:

            model_path_base = self.model_path_name + str(self.epochs_total)

            # model.save( model_path_base + '_' + str(acc) ,True, True)

            weights = model_path_base + '_' + str(tested_acc) + '_weights'
            self.model.save_weights(weights)

            # if acc < savedAcc - 0.1:  # REMOVE
            #     model.save_weights(os.path.join(config['data_path'],'CHECK'+str(epochs_total)))
            # if acc> savedAcc:
            #     savedAcc = acc


            d = {'accuracy': self.accs, 'history': self.hist, 'epochs_total': self.eps_tot}

            losses = self.model_path_name + '_losses'

            with open(losses + '_tmp', 'wb') as handle:
                pickle.dump(d, handle)

            if os.path.exists(losses):
                os.remove(losses)

            os.rename(losses + '_tmp', losses)
            if not self.config['spareWeightFilesFunc'](self.epochs_total):
                self.savedFiles.append(weights)
            if len(self.savedFiles) > 3:
                for filename in glob.glob(self.savedFiles[0] + "*"):
                    try:
                        os.remove(filename)
                    except IsADirectoryError:
                        shutil.rmtree(filename)
                del self.savedFiles[0]

    def evaluate(self, X, genos_true):
        res = self.model.predict(X, batch_size=self.config['batch_size'])

        genosCheck = self.get_genos(res)

        sumOfTokens = np.sum(X[1])

        hits = np.where((X[1] == 1) & (genos_true == genosCheck), 1, 0)
        sumOfHits = hits.sum()

        acc = sumOfHits / sumOfTokens

        print("Token accuracy: " + str(acc) + ", " + self.config['model_name'])

        return acc

    def fill_geno_bank(self, need_value = None):
        print("Filling geno bank")
        if not self.cli_orchestra.initialized:
            self.cli_orchestra.create_orchestra()

        if len(self.starting_genos) == 0:
            self.simplest = self.cli_orchestra.get_simplest(self.config['representation'][-1:]).result()
        self.starting_genos.append(self.simplest)
        # try:
        # FramsticksCLI.PRINT_FRAMSTICKS_OUTPUT = True
        genos_batches = [self.cli_orchestra.ask_for_genos(self.starting_genos, self.config['generate_size'], self.config['diversity'], timeout=self.config['generate_timeout']) for i in range(self.cli_orchestra.workers)]


        # except:
        #     self.createCLI(level=1)
        #     self.fill_geno_bank(need_value)
        #     return
        for res in genos_batches:
            geno_batch = res.result()
            if geno_batch is not None:
                for geno in geno_batch:
                    self.geno_bank[len(geno)].append(geno)

        added_genos = 0
        attempts = 0
        for k, v in self.geno_bank.items():
            print(k, len(v))

        if need_value is not None:
            mink, maxk = self.get_close_range(need_value)
            for j in range(self.config['need_value_proximity_times']):
                for i in range(mink, maxk):
                    self.add_random_of_a_given_len_to_starting_points(i)



        while added_genos < self.config['starting_points_batch'] and attempts<self.config['starting_add_attempts']:
            leng = self.config['starting_point_len_distribution']()
            if self.add_random_of_a_given_len_to_starting_points(leng):
                added_genos += 1
            attempts += 1





    def add_random_of_a_given_len_to_starting_points(self,k):
        v = self.geno_bank[k]
        if len(v) > 0:
            geno = np.random.choice(v)
            if geno not in self.starting_genos:
                self.starting_genos.append(geno)
                return True
        return False
    def get_close_range(self, k):
        mink = max(1, k - self.config['close_range'])
        maxk = min(self.config['max_len'] - 1, k + self.config['close_range'] + 1)
        return mink, maxk
    def get_geno_from_bank(self, k):

        # mink, maxk = self.get_close_range(k)
        # len_range = np.arange(mink, maxk)
        # k_ind = np.argwhere(len_range==k)[0][0]


        # p = stats.norm.pdf(len_range, loc=k, scale=np.std(len_range)/1.4)
        # lenp = np.array([len(self.geno_bank[q]) for q in len_range])
        # meanlenp = np.sum(lenp)
        # lenpp = p+(p*meanlenp**0.7)+p*lenp
        # lenpp /= np.sum(lenpp)
        # # lenpp[k_ind] += 0.25
        #
        # k = np.random.choice(len_range, p=lenpp)

        genos = self.geno_bank[k]
        if len(genos) == 0:
            # print(k)
            self.fill_geno_bank(k)
            return self.get_geno_from_bank(k)
        return genos.pop()


    def get_genos(self, dec_predict, return_discrete = False):
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
        if return_discrete:
            return genosCheck, resMax
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


    def decode_latents(self, latents : np.ndarray) -> List[str]:
        hidden, cell = splitLatentToHiddenAndCell(latents)
        # hidden = np.clip(hidden, a_min=-1, a_max=1)
        pred = self.decoder.predict(
            [hidden, cell, self.getOnesMaskForFullPrediction(len(latents))])
        genos = self.get_genos(pred)

        frams = [extract_fram(geno) for geno in genos]
        return frams

    def sampleMultivariateGaussianLatent(self, samples, power=1.0):
        sampled = np.random.multivariate_normal(self.means, self.cov*power, size=samples)

        return sampled

    def prepareLatentProperties(self):
        if self.means_mut is None:
            self.means_mut = np.zeros((self.config['cells']*2,))

        cov_path = self.model_path_name+"_cov"
        means_path = self.model_path_name+"_means"
        latent_path =  self.model_path_name+"_latent"
        if os.path.exists(cov_path):
            if self.clear_files:
                os.remove(cov_path)
            else:
                with open(cov_path, 'rb') as handle:
                    self.cov = pickle.load(handle)
                    print("Loaded cov")
        if os.path.exists(means_path):
            if self.clear_files:
                os.remove(means_path)
            else:
                with open(means_path, 'rb') as handle:
                    self.means = pickle.load(handle)
                    print("Loaded means")

        if os.path.exists(latent_path):
            if self.clear_files:
                os.remove(latent_path)
            else:
                with open(latent_path, 'rb') as handle:
                    self.latent = pickle.load(handle)
                    print("Loaded latent")


        if self.cov is None or self.means is None or self.latent is None:
            self.latent = self.predictEncoderLatent(self.data[0])
            self.means = np.mean(self.latent, axis=0)
            self.cov = np.cov(self.latent.T)
            print("Calculated latens, means and cov")
            with open(cov_path, 'wb') as handle:
                pickle.dump(self.cov, handle)
            with open(means_path, 'wb') as handle:
                pickle.dump(self.means, handle)
            with open(latent_path, 'wb') as handle:
                pickle.dump(self.latent, handle)


    def load_model_data(self):
        losses_path = None
        expected_epochs = None
        # losses_pathFitn = None
        # expected_epochsFitn = None

        print(self.model_path_name)
        if os.path.exists(self.model_path_name + '_losses' + '_tmp'):
            losses_path = self.model_path_name + '_losses' + '_tmp'
        else:
            if os.path.exists(self.model_path_name + '_losses'):
                losses_path = self.model_path_name + '_losses'

        # if os.path.exists(self.fitnessModel_path_name + '_losses' + '_tmp'):
        #     losses_pathFitn = self.fitnessModel_path_name + '_losses' + '_tmp'
        # else:
        #     if os.path.exists(self.fitnessModel_path_name + '_losses'):
        #         losses_pathFitn = self.fitnessModel_path_name + '_losses'
        #
        # if losses_pathFitn is not None:
        #     with open(losses_pathFitn, 'rb') as handle:
        #         losses_fileFitn = pickle.load(handle)
        #     expected_epochsFitn = losses_fileFitn['epochs_total'][-1]

        # if expected_epochsFitn is not None:
        #     files = []
        #     for filename in glob.glob(self.fitnessModel_path_name + "*"):
        #         files.append('.'.join(filename.split('.')[:-1]))
        #
        #     files = list(set(files))
        #
        #     if len(files) > 0:
        #         files = sorted([(f, int(f[len(self.fitnessModel_path_name):].split('_')[0])) for f in files if
        #                         RepresentsInt(f[len(self.fitnessModel_path_name):].split('_')[0])], key=lambda x: x[1])
        #
        #         break_ind = -1
        #         if expected_epochs is not None:
        #             for ind, f in enumerate(files):
        #                 if f[1] == expected_epochs:
        #                     break_ind = ind
        #
        #         files = [f for f, _ in files]
        #
        #         self.savedFilesFitn = files[:break_ind + 1]
        #
        #         # for fDel in [f for f in files if f not in savedFiles]:
        #         #     for filename in glob.glob(fDel+"*"):
        #         #         print("Deleting not matching weights file: " + str(filename))
        #         #         os.remove(filename)
        #
        #         if len(self.savedFilesFitn) > 0:
        #             print("Loaded saved files: " + str(self.savedFilesFitn))
        #             load_weights(self.fitnessModel, None, self.savedFilesFitn[-1])
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

        if self.config['requested_epoch'] is not None:
            expected_epochs = self.config['requested_epoch']

        if expected_epochs is not None:
            self.config['past_epochs'] = expected_epochs
            self.config['lr'] *= self.config['lr_epochs_done_decay'] ** expected_epochs
            self.config['lr'] = max(self.config['lr'], self.config['min_lr'])
            print("Using learning rate: ", self.config['lr'])

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
            if break_ind == -1 and self.config['let_not_newest_weights']:
                break_ind=len(files)-1

            self.savedFiles = [f for f, eps in files if (not self.config['spareWeightFilesFunc'](eps)) and (not (self.config['requested_epoch'] == eps)) ]
            files = [f for f, _ in files]
            print("Stored weight files: " + str(files))
            # = files#files[break_ind] #[:break_ind + 1]

            # for fDel in [f for f in files if f not in savedFiles]:
            #     for filename in glob.glob(fDel+"*"):
            #         print("Deleting not matching weights file: " + str(filename))
            #         os.remove(filename)

            if break_ind >= 0:
                load_f = files[break_ind]
                load_weights(self.model, self.decoder, load_f)
            else:
                print("Did not find matching save file to last loss or requested epoch!")
                self.hist = []
                self.accs = []
                self.eps_tot = []
        else:
            self.savedFiles = []

        if expected_epochs is not None:
            self.config['past_epochs'] = expected_epochs

from scipy.spatial.distance import pdist, squareform
from textdistance import levenshtein
from threading import Lock
from functools import partial
from concurrent.futures import ThreadPoolExecutor
from time import sleep
custom_levens = lambda x,y : levenshtein(x[0], y[0])

def get_genos_dist(genos):
    XA = np.reshape(genos, (-1,1))
    return squareform(pdist(XA, custom_levens))
def dissimilarity_lev(genos):
    return get_genos_dist(genos)

def dissimilarity_rand(genos):
    return np.random.rand(len(genos), len(genos))
#####################

class CLIHandler():

    def __init__(self, config, pid_extra_extension : str):
        self.config = {}
        self.framsCLI : FramsticksCLI= None
        self.invalidCnt = {}
        self.validCnt = {}
        self.pid_extra_extension = pid_extra_extension
        self.dissimilarity = None
        self.busy = False
        self.pid_version = -1
        for k in ['prefix', 'frams_path', 'framsexe', 'model_name', 'pid_extension', 'importSim', 'markers', 'representation', 'locality_type']:
            if k in config:
                self.config[k] = config[k]


    def createCLI(self, level=0):
        self.pid_version += 1
        if level > 20:
            raise ConnectionError("FramsCLI recursion over 20")
        try:
            if self.framsCLI is not None and level < 10:
                self.framsCLI.closeFramsticksCLI()
            framsCLId = FramsticksCLI(self.config['frams_path'], None if 'framsexe' not in self.config else self.config['framsexe'],
                                      pid=self.config['model_name'] + "_" + str(level) + (
                                          '' if 'pid_extension' not in self.config else self.config['pid_extension']) +
                                       "_" + self.pid_extra_extension + "_" + str(self.pid_version),
                                      importSim=self.config['importSim'], markers=self.config['markers'], config=self.config)
            print("OK_CORRECTLY CREATED__", level)
        except Exception as e:
            self.createCLI(level=level + 1)
            return
        self.framsCLI = framsCLId
        self.framsCLI.invalidCnt = self.invalidCnt
        self.framsCLI.validCnt = self.validCnt
        self.framsCLI.em = self
        if self.config['locality_type'] == 'dissim' or self.config['locality_type'] == 'fitness':
            self.dissimilarity = self.framsCLI.dissimilarity
        elif self.config['locality_type'] == 'levens':
            self.dissimilarity = dissimilarity_lev
        else:
            self.dissimilarity = dissimilarity_rand

def clihandler_runf(clihandler : CLIHandler, func, lock : Lock):
    try:
        ret = func(clihandler)
        # print(ret)
        return ret
    except Exception as e:
        clihandler.createCLI(1)
        return None
    finally:
        lock.acquire()
        clihandler.busy = False
        lock.release()



class CLIOrchestra():
    def __init__(self, config):
        self.config = config
        self.workers = self.config['locality_prepare_workers']
        self.handlers = [CLIHandler(config, str(i)) for i in range(self.workers)]
        self.lock = Lock()
        self.initialized = False
    def create_orchestra(self):
        self.initialized = True
        self.e = ThreadPoolExecutor(max_workers=self.workers)
        for handler in self.handlers:
            handler.createCLI()
    def run_batch(self, func, changing_params_list, kwargs):
        if isinstance(changing_params_list, list):
            runs_args=[{**kwargs, **params} for params in changing_params_list]
        else:
            runs_args = [kwargs for i in range(changing_params_list)]
        return [self.run_task(func, **args) for args in runs_args]

    def get_simplest(self, genetic_format):
        chosen_cli = self.find_not_busy()
        f = lambda cli_handler : FramsticksCLI.getSimplest(cli_handler.framsCLI, genetic_format)
        # cli_f = partial(f, chosen_cli, genetic_format)
        return self.e.submit(clihandler_runf, chosen_cli, f, self.lock)
    def ask_for_genos(self, starting_genos, generate_size, diversity, timeout=None):
        chosen_cli = self.find_not_busy()
        f = lambda cli_handler: FramsticksCLI.ask_for_genos(cli_handler.framsCLI,starting_genos, generate_size, diversity, timeout)
        # cli_f = partial(FramsticksCLI.ask_for_genos, chosen_cli, starting_genos, generate_size, diversity, timeout)
        return self.e.submit(clihandler_runf, chosen_cli, f, self.lock)


    def find_not_busy(self):
        chosen_cli = None
        cli_ind = -1
        while True:
            cli_ind += 1
            if cli_ind >= self.workers:
                cli_ind = 0
                sleep(0.05)
            if not self.handlers[cli_ind].busy:
                chosen_cli = self.handlers[cli_ind]
                self.lock.acquire()
                chosen_cli.busy = True
                self.lock.release()
                break
        return chosen_cli
    def run_task(self, func, **kwargs):
        chosen_cli = self.find_not_busy()
        # print("running", i, len(genos_batch))
        cli_f = partial(func, **kwargs)
        return self.e.submit(clihandler_runf, chosen_cli, cli_f, self.lock)


####################################


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



def checkValid(evolution_model : EvolutionModel, frams, level=0, valid_key=None):
    # evolution_model.invalidCnt = evolution_model.framsCLI.invalidCnt
    # evolution_model.validCnt = evolution_model.framsCLI.validCnt
    try:

        return evolution_model.framsCLI.isValid([evolution_model.config['prefix'] + fram for fram in frams], valid_key)
    except:
        print("Error, checkValid, level ", level)
        evolution_model.createCLI(level = level)
        return checkValid(evolution_model, frams, level+1, valid_key)

def evaluate_with_batches(evolution_model : EvolutionModel, frams, base_fitness = None, return_inds = False, return_bad = False, ind_base = None, eval_single=True, max_at_once=50):
    # max_at_once = 50

    fitnesses = []
    good_indss = []
    bad_mutants = 0
    if ind_base is None:
        ind_base = np.arange(len(frams))
    for i in range(0, len(frams), max_at_once):
        temp = frams[i:i + max_at_once]
        temp = [(evolution_model.config['prefix'] + tem) if not tem.startswith(evolution_model.config['prefix']) else tem for tem in temp]

        try:
            eval_result = evolution_model.framsCLI.evaluate(tuple(temp))
            fitness, bad_mutants_temp, good_inds = extractFitnesses(eval_result, list(ind_base[list(range(i, i + len(temp)))]))
            bad_mutants += bad_mutants_temp
            good_indss.extend(good_inds)
        except:
            evolution_model.createCLI()
            fitness = []
            if eval_single:
                for q in range(len(temp)):
                    t = (temp[q],)
                    try:
                        fitnPart_eval = evolution_model.framsCLI.evaluate(t)
                        fitnPart, bad_mutants_temp, good_ind = extractFitnesses(fitnPart_eval, [ind_base[i + q]])

                        if bad_mutants_temp == 0:
                            fitness.append(fitnPart[0])
                            good_indss.extend(good_ind)
                        else:
                            bad_mutants +=1
                    except:
                        evolution_model.createCLI()
        fitnesses.extend(fitness)

    if base_fitness is not None:
        fitnesses = [base_fitness if i not in good_indss else fitnesses[good_indss.index(i)] for i in range(len(frams))]

    if return_bad:
        return fitnesses, good_indss, bad_mutants
    if return_inds:
        return fitnesses, good_indss
    else:
        return fitnesses


def evaluate_with_batches2(framsCLIHandler : CLIHandler, frams=None, base_fitness = None, return_inds = False, return_bad = False, ind_base = None, eval_single=True, max_at_once=50):
    # max_at_once = 50
    if frams is None:
        raise Exception("No frams")
    fitnesses = []
    good_indss = []
    bad_mutants = 0
    if ind_base is None:
        ind_base = np.arange(len(frams))
    for i in range(0, len(frams), max_at_once):
        temp = frams[i:i + max_at_once]
        temp = [(framsCLIHandler.config['prefix'] + tem) if not tem.startswith(framsCLIHandler.config['prefix']) else tem for tem in temp]

        try:
            eval_result = framsCLIHandler.framsCLI.evaluate(tuple(temp))
            fitness, bad_mutants_temp, good_inds = extractFitnesses(eval_result, list(ind_base[list(range(i, i + len(temp)))]))
            bad_mutants += bad_mutants_temp
            good_indss.extend(good_inds)
        except:
            framsCLIHandler.createCLI()
            fitness = []
            if eval_single:
                for q in range(len(temp)):
                    t = (temp[q],)
                    try:
                        fitnPart_eval = framsCLIHandler.framsCLI.evaluate(t)
                        fitnPart, bad_mutants_temp, good_ind = extractFitnesses(fitnPart_eval, [ind_base[i + q]])

                        if bad_mutants_temp == 0:
                            fitness.append(fitnPart[0])
                            good_indss.extend(good_ind)
                        else:
                            bad_mutants +=1
                    except:
                        framsCLIHandler.createCLI()
        fitnesses.extend(fitness)

    if base_fitness is not None:
        fitnesses = [base_fitness if i not in good_indss else fitnesses[good_indss.index(i)] for i in range(len(frams))]

    if return_bad:
        return fitnesses, good_indss, bad_mutants
    if return_inds:
        return fitnesses, good_indss
    else:
        return fitnesses