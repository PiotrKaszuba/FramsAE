import random
import re
from typing import Tuple, List, Callable

import numpy as np
from collections import deque
from Code.Preparation.ParseGenFile import testGenos
from Code.Preparation.encodeGenos import encodeGenos
import tensorflow as tf

pad_sequences = tf.keras.preprocessing.sequence.pad_sequences



def RepresentsInt(s):
    try:
        int(s)
        return True
    except ValueError:
        return False

def get_genSuff(representation):
    if representation == 'f4':
        return '/*4*/'
    if representation == 'f9':
        return '/*9*/'
    return ''


def extract_fram(fram):
    fram = ''.join(list(fram))
    s = re.search('S[^ST]*T', fram)
    if s:
        fram = s.group()

    fram = re.sub('[ST]', '', fram)

    return fram

from Code.FramsticksCli import fake_fitness
from functools import partial

def spareWeightFunc(epoch):
    return (epoch == 631) or ((epoch % 50 == 1) and epoch > 3)  # lambda epoch: False
def probe_distribution_f(lengths, probs):
    k = np.random.choice(lengths, p=probs)
    return k

def make_probe_distribution_f(lengths, probs):
    print("distribution being made: ", probs)
    return partial(probe_distribution_f, lengths, probs)

def make_lengths(min, max):
    lengths = np.arange(min, max + 1)
    return lengths

def exponential_distribution(min, max, a, b, base):
    # base**(ax+1)+b
    lengths = np.arange(min, max+1)
    probs = base**(a*lengths+1) + b
    probs = probs / np.sum(probs)
    return probs
def starting_points_distribution(min, max, a, b, c, poly):
    probs = polynomial_distribution(min, max, a, b, c, poly)
    probs = probs * probs * probs[::-1]
    probs = probs / np.sum(probs)
    return probs

def polynomial_distribution(min, max, a ,b, c, poly):
    # (ax + b)**poly + c
    lengths = np.arange(min, max+1)

    probs = ((a*lengths+b) ** poly) + c
    probs = probs/np.sum(probs)
    return probs


def extractFitnesses(eval_result, inds):
    fitnesses = []
    inds_out = []
    bad_mutants_f = 0
    for res,ind in zip(eval_result, inds):
        try:
            if not fake_fitness[0]:
                fitness = float(res['evaluations']['']['vertpos'])
                fitness *= float(res['fitn_mult'])
                fitness += float(res['len_fitn'])
            else:
                fitness = float(res)
            if fitness < res['min_fitn']:
                fitness = res['min_fitn']
            fitnesses.append(fitness)
            inds_out.append(ind)
        except:
            bad_mutants_f +=1
    return fitnesses, bad_mutants_f, inds_out


class DatasetProvider:
    train_data = None
    get_train_objects = None
    test_data = None
    get_test_objects = None

    def __init__(self, train_data_get_methods_maker, test_data_get_methods_maker, config, **kwargs):
        self.train_methods = train_data_get_methods_maker(config, **kwargs)
        self.test_methods = test_data_get_methods_maker(config, **kwargs)

    def get_data(self, howMuch=None) -> Tuple[List, Callable]:
        pass
    def get_test_data(self, howMuch=None) -> Tuple[List, Callable]:
        pass

    def get_ask_for_objects_f(self, data, get_for_objects_func):
        return lambda x, start=0, end=None, indices=None: get_for_objects_func(data, x, start, end, indices)

class OneCallDatasetProvider(DatasetProvider):

    def __init__(self, train_data_get_methods_maker, test_data_get_methods_maker, config, **kwargs):
        DatasetProvider.__init__(self, train_data_get_methods_maker, test_data_get_methods_maker, config, **kwargs)

        self.train_data = self.train_methods[0]()
        self.get_train_objects = self.train_methods[1]
        self.test_data = self.test_methods[0]()
        self.get_test_objects = self.test_methods[1]

    def get_data(self, howMuch=None) -> Tuple[List, Callable]:
        if howMuch is None:
            howMuch = len(self.test_data[0])
        data = list(zip(*self.train_data))
        random.shuffle(data)
        data = tuple(np.array(elem)[:howMuch] if isinstance(elem[0], np.ndarray) else list(elem)[:howMuch] for elem in (zip(*data)))

        return data, lambda x, start=None, end=None: self.get_train_objects(data, x, start, end)

    def get_test_data(self, howMuch=None) -> Tuple[List, Callable]:
        if howMuch is None:
            howMuch = len(self.test_data[0])

        data = list(zip(*self.test_data))
        random.shuffle(data)
        data = tuple(np.array(elem)[:howMuch] if isinstance(elem[0], np.ndarray) else list(elem)[:howMuch] for elem in (zip(*data)))


        return data, lambda x, start=None, end=None: self.get_test_objects(data, x, start, end)


class DequeMultipleCallDatasetProvider(DatasetProvider):

    def __init__(self, train_data_get_methods_maker, test_data_get_methods_maker, config, **kwargs):
        DatasetProvider.__init__(self, train_data_get_methods_maker, test_data_get_methods_maker, config, **kwargs)
        self.train_data = deque(maxlen=config['train_deque_size'])
        self.test_data = deque(maxlen=config['test_deque_size'])
        self.train_data_feed = self.train_methods[0]
        self.test_data_feed = self.test_methods[0]

        while(len(self.train_data) < config['train_deque_size']):
            self.train_data.extend(list(zip(*self.train_data_feed())))

        while (len(self.test_data) < config['test_deque_size']):
            self.test_data.extend(list(zip(*self.test_data_feed())))

        self.get_train_objects = self.train_methods[1]
        self.get_test_objects = self.test_methods[1]

    def get_data(self, howMuch=None) -> Tuple[List, Callable]:
        if howMuch is None:
            howMuch = len(self.train_data)
        data = random.sample(self.train_data, k=howMuch)
        data = tuple(np.array(elem)[:howMuch] if isinstance(elem[0], np.ndarray) else list(elem)[:howMuch] for elem in (zip(*data)))
        self.train_data.extend(list(zip(*self.train_data_feed())))
        return data, lambda x, start=None, end=None: self.get_train_objects(data, x, start, end)

    def get_test_data(self, howMuch=None) -> Tuple[List, Callable]:
        # if howMuch is None:
        howMuch = len(self.test_data)

        data = random.sample(self.test_data, k=howMuch)
        data = tuple(np.array(elem)[:howMuch] if isinstance(elem[0], np.ndarray) else list(elem)[:howMuch] for elem in (zip(*data)))

        self.test_data.extend(list(zip(*self.test_data_feed())))
        return data, lambda x, start=None, end=None: self.get_test_objects(data, x, start, end)



def prepareDataset(genos, config, **kwargs):
    ([X, zerosInputs], Y, genosCheck) = prepareGenos(genos, config)
    if config['locality_term']:
        howMuchData = (len(genos) // config['batch_size']) * config['batch_size']
    else:
        howMuchData = len(genos)

    X, zerosInputs, Y, genosCheck, genos = [dat[:howMuchData] for dat in [X, zerosInputs, Y, genosCheck, genos]]
    return X, zerosInputs, Y, genosCheck, genos


def get_train_set_func_make(config, **kwargs):
    def get_train_set_func():
        genos = testGenos(config, print_some_genos=True)
        out = prepareDataset(genos, config)
        return out

    def get_objects(data, name, start=0, end=None, indices=None):
        if indices is None:
            if end is None:
                end = len(data[4])
            if start is None:
                start = 0
            indices = np.arange(start, end)

        if name == 'input':
            return [data[0][indices], data[1][indices]]
        elif name == 'output':
            return data[2][indices]
        elif name=='genosCheck':
            return data[3][indices]
        elif name=='genos':
            return list(np.array(data[4])[indices])
        else:
            res = []
            for ind, dat in enumerate(data):
                if ind == 4:
                    obj = list(np.array(dat)[indices])
                else:
                    obj = dat[indices]
                res.append(obj)
            return res

    return get_train_set_func, get_objects

def random_f9(config, evolution_model, add_ST=True, letters=None):
    if letters is None:
        letters = config['dict'].replace('S','').replace('T','')



    # lengths = np.arange(config['max_len']-2)+1
    # k = np.random.choice(lengths, p=lengths/np.sum(lengths))
    k = config['geno_len_distribution']()
    # print(k)
    geno = ''.join(random.choices(letters, k=k))
    if add_ST:
        geno = 'S'+ geno +'T'
    return geno

def get_train_set_func_f9_stream_make(config, **kwargs):
    em = kwargs['em']
    letters = config['dict'].replace('S','').replace('T','')
    def get_train_set_func():
        genos =  [random_f9(config, em, letters=letters) for i in range(config['train_deque_size'])]
        out = prepareDataset(genos, config)
        return out

    def get_objects(data, name, start=0, end=None, indices=None):
        if indices is None:
            if end is None:
                end = len(data[4])
            if start is None:
                start = 0
            indices = np.arange(start, end)

        if name == 'input':
            return [data[0][indices], data[1][indices]]
        elif name == 'output':
            return data[2][indices]
        elif name=='genosCheck':
            return data[3][indices]
        elif name=='genos':
            return list(np.array(data[4])[indices])
        else:
            res = []
            for ind, dat in enumerate(data):
                if ind == 4:
                    obj = list(np.array(dat)[indices])
                else:
                    obj = dat[indices]
                res.append(obj)
            return res

    return get_train_set_func, get_objects

def get_test_set_func_f9_stream_make(config, **kwargs):
    em = kwargs['em']
    letters = config['dict'].replace('S', '').replace('T', '')
    def get_test_set_func():
        genos =  [random_f9(config, em, letters=letters) for i in range(config['batch_size'])]
        X, zerosInputs, Y, genosCheck, genos = prepareDataset(genos, config)
        if config['locality_term']:
            locality_vals = np.zeros((len(genos), config['batch_size'], config['batch_size']))
            return X, zerosInputs, Y, genosCheck, genos, locality_vals
        return X, zerosInputs, Y, genosCheck, genos


    def get_objects(data, name, start=0, end=None, indices=None):
        if indices is None:
            if end is None:
                end = len(data[4])
            if start is None:
                start = 0
            indices = np.arange(start, end)

        if name == 'input':
            if config['locality_term']:
                return [data[0][indices], data[1][indices], data[5][indices]]
            else:
                return [data[0][indices], data[1][indices]]
        elif name == 'output':
            return data[2][indices]
        elif name=='genosCheck':
            return data[3][indices]
        elif name=='genos':
            return list(np.array(data[4])[indices])
        else:
            res = []
            for ind, dat in enumerate(data):
                if ind == 4:
                    obj = list(np.array(dat)[indices])
                else:
                    obj = dat[indices]
                res.append(obj)
            return res

    return get_test_set_func, get_objects

def random_f1(config, evolution_model, add_ST=True):
    # letters = config['dict'].replace('S', '').replace('T', '')
    # lengths = np.arange(config['max_f1_len']) + 1
    #
    # # lengths = np.arange(config['max_len'] - 2) + 1
    # ones = np.ones_like(lengths)
    # # lengths_norm = ones/np.sum(ones)
    #
    # lengths_norm = lengths/np.sum(lengths)
    # k = np.random.choice(lengths, p=lengths_norm)
    k = config['geno_len_distribution']()
    geno = evolution_model.get_geno_from_bank(k)
    # print('1', geno)
    # print(geno)
    # geno = 'X'

    geno = re.sub('[^' + config['dict'] + ']', '', geno)
    # print('2', geno)
    if add_ST:
        geno = 'S'+ geno +'T'
    return geno


def get_train_set_func_f1_stream_make(config, **kwargs):
    em = kwargs['em']
    def get_train_set_func():
        genos =  [random_f1(config, em) for i in range(config['train_deque_size'])]
        out = prepareDataset(genos, config)
        return out

    def get_objects(data, name, start=0, end=None, indices=None):
        if indices is None:
            if end is None:
                end = len(data[4])
            if start is None:
                start = 0
            indices = np.arange(start, end)

        if name == 'input':
            return [data[0][indices], data[1][indices]]
        elif name == 'output':
            return data[2][indices]
        elif name=='genosCheck':
            return data[3][indices]
        elif name=='genos':
            return list(np.array(data[4])[indices])
        else:
            res = []
            for ind, dat in enumerate(data):
                if ind == 4:
                    obj = list(np.array(dat)[indices])
                else:
                    obj = dat[indices]
                res.append(obj)
            return res

    return get_train_set_func, get_objects

def get_test_set_func_f1_stream_make(config, **kwargs):
    em = kwargs['em']
    def get_test_set_func():
        genos =  [random_f1(config, em) for i in range(config['batch_size'])]
        X, zerosInputs, Y, genosCheck, genos = prepareDataset(genos, config)
        if config['locality_term']:
            locality_vals = np.zeros((len(genos), config['batch_size'], config['batch_size']))
            return X, zerosInputs, Y, genosCheck, genos, locality_vals
        return X, zerosInputs, Y, genosCheck, genos

    def get_objects(data, name, start=0, end=None, indices=None):
        if indices is None:
            if end is None:
                end = len(data[4])
            if start is None:
                start = 0
            indices = np.arange(start, end)

        if name == 'input':
            if config['locality_term']:
                return [data[0][indices], data[1][indices], data[5][indices]]
            else:
                return [data[0][indices], data[1][indices]]
        elif name == 'output':
            return data[2][indices]
        elif name=='genosCheck':
            return data[3][indices]
        elif name=='genos':
            return list(np.array(data[4])[indices])
        else:
            res = []
            for ind, dat in enumerate(data):
                if ind == 4:
                    obj = list(np.array(dat)[indices])
                else:
                    obj = dat[indices]
                res.append(obj)
            return res

    return get_test_set_func, get_objects



def get_test_set_func_make(config, **kwargs):
    def get_test_set_func():
        temp = config['long_genos']
        config['long_genos'] = 'TEST'
        genos = testGenos(config, print_some_genos=True)
        X, zerosInputs, Y, genosCheck, genos = prepareDataset(genos, config)
        config['long_genos'] = temp
        if config['locality_term']:
            locality_vals = np.zeros((len(genos), config['batch_size'], config['batch_size']))
            return X, zerosInputs, Y, genosCheck, genos, locality_vals
        return X, zerosInputs, Y, genosCheck, genos

    def get_objects(data, name, start=0, end=None, indices=None):
        if indices is None:
            if end is None:
                end = len(data[4])
            if start is None:
                start = 0
            indices = np.arange(start, end)

        if name == 'input':
            if config['locality_term']:
                return [data[0][indices], data[1][indices], data[5][indices]]
            else:
                return [data[0][indices], data[1][indices]]
        elif name == 'output':
            return data[2][indices]
        elif name=='genosCheck':
            return data[3][indices]
        elif name=='genos':
            return list(np.array(data[4])[indices])
        else:
            res = []
            for ind, dat in enumerate(data):
                if ind == 4:
                    obj = list(np.array(dat)[indices])
                else:
                    obj = dat[indices]
                res.append(obj)
            return res

    return get_test_set_func, get_objects


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



def prepareData(config, ret_pure_genos = False):
    genos = testGenos(config, print_some_genos=True)
    if ret_pure_genos:
        return prepareGenos(genos, config), genos
    return prepareGenos(genos, config)


def zscore(x):
  mean = tf.reduce_mean(x, axis=0)
  std = tf.math.reduce_std(x, axis=0)
  return (x-mean)/(std+0.01)


def tf_corr(x,y):
    num = tf.reduce_sum(tf.cast(x, tf.float32) * tf.cast(y, tf.float32)) - tf.cast(tf.shape(x)[0],
                    tf.float32) * tf.cast(tf.reduce_mean(x), tf.float32) * tf.cast(tf.reduce_mean(y), tf.float32)

    den = tf.cast(tf.shape(x)[0], tf.float32) * tf.cast(tf.math.reduce_std(x), tf.float32) * tf.cast(tf.math.reduce_std(y), tf.float32)
    return num/(den+0.01)


def squared_dist(A):
    expanded_a = tf.expand_dims(A, 1)
    expanded_b = tf.expand_dims(A, 0)
    distances = tf.reduce_sum(tf.math.squared_difference(expanded_a, expanded_b), 2)
    return distances


def euclidean(A):
    sq_dist = tf.cast(squared_dist(A), tf.float32)
    return tf.where(sq_dist != 0, tf.sqrt(sq_dist), sq_dist)


def upper_triangular(A):
    ones = tf.ones_like(A)
    mask_a = tf.linalg.band_part(ones, 0, -1)  # Upper triangular matrix of 0s and 1s
    mask_b = tf.linalg.band_part(ones, 0, 0)  # Diagonal matrix of 0s and 1s
    mask = tf.cast(mask_a - mask_b, dtype=tf.bool)  # Make a bool mask

    upper_triangular_flat = tf.boolean_mask(A, mask)
    return upper_triangular_flat


def locality1_op(x):
    latent = zscore(tf.concat([x[0], x[1]], axis=-1))
    latent_dist = upper_triangular(euclidean(latent))
    return latent_dist


def locality2_op(x):
    genos_dist = upper_triangular(x[0])
    return genos_dist

@tf.function
def locality_term_op(x):
    corr = tf_corr(x[0], x[1])
    return corr


