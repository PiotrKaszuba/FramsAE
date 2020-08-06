import tensorflow as tf
import random
from Code.FramsticksCli import FramsticksCLI
import numpy as np
from collections import defaultdict
from scipy.spatial.distance import pdist, squareform

from textdistance import levenshtein
custom_levens = lambda x,y : levenshtein(x[0], y[0])

def get_genos_dist(genos):
    XA = np.reshape(genos, (-1,1))
    return squareform(pdist(XA, custom_levens))


class LevenshteinDissim:

    def dissimilarity(self, genos):
        return get_genos_dist(genos)

class NoLocDissim:

    def dissimilarity(self, genos):
        return np.random.rand(len(genos), len(genos))

def zscore(x):
  mean = tf.reduce_mean(x, axis=0)
  std = tf.math.reduce_std(x, axis=0)
  return (x-mean)/std

def tf_corr(x,y):
    num = tf.reduce_sum(x * y ) - tf.cast(tf.shape(x)[0], tf.float32) * tf.reduce_mean(x) * tf.reduce_mean(y)
    den = tf.cast(tf.shape(x)[0], tf.float32) * tf.math.reduce_std(x) * tf.math.reduce_std(y)
    return num/den


def squared_dist(A):
    expanded_a = tf.expand_dims(A, 1)
    expanded_b = tf.expand_dims(A, 0)
    distances = tf.reduce_sum(tf.math.squared_difference(expanded_a, expanded_b), 2)
    return distances

def euclidean(A):
    return tf.sqrt(tf.cast(squared_dist(A), tf.float32))

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

def locality_term_op(x):
    corr = tf_corr(x[0], x[1])
    return corr

def merge_histories(histories):
    d = defaultdict(list)
    for hist in histories:
        for key, value in hist.items():
            d[key].extend(value)
    return d


def createCLI(config, oldCLI = None, level = 0):
    if level > 20:
        raise ConnectionError("FramsCLI recursion over 20")
    try:
        if oldCLI is not None and level < 10:
            oldCLI.closeFramsticksCLI()
        framsCLId = FramsticksCLI(config['frams_path'], None, pid=config['model_name'])
        framsCLId.rawCommand("Simulator.print(ModelSimilarity.simil_partdeg=1.0);", marker="Script.Message")
        framsCLId.rawCommand("Simulator.print(ModelSimilarity.simil_partgeom=1.0);", marker="Script.Message")
    except:
        return createCLI(config, oldCLI, level= level+1)
    return framsCLId

def train_with_locality(model, data, config, valid_data):
    if config['locality_type'] == 'dissim':
        framsCLI = createCLI(config)
    elif config['locality_type'] == 'levens':
        framsCLI = LevenshteinDissim()
    else:
        framsCLI = NoLocDissim()
    X_train, zerosTrain, genos, y_train = data
    epochs = config['epochs_per_i']
    histories = []
    # d_errs_sum = Counter()

    # if os.path.exists('d_errs_sum'):
    #     with open('d_errs_sum', 'rb') as f:
    #         d_errs_sum = pickle.load(f)
    for ep in range(epochs):
        history = fit_epoch(model, [X_train, zerosTrain, genos, y_train], config, valid_data, framsCLI)
        # d_errs_sum = sum([d_errs_sum, d_errs], Counter())
        # print(d_errs_sum)
        histories.append(history)
        # to_pop = []
        # for i in range(len(genos)):
        #     if d_errs_sum[genos[i]] == 2:
        #         print("Dangerous (2): ", genos[i])
        #     if d_errs_sum[genos[i]] > 2:
        #         to_pop.append(i)
        #         print("Popping (3): ", genos[i])



        # genos = list(np.delete(genos, to_pop, axis=0))
        # X_train = np.delete(X_train, to_pop, axis=0)
        # zerosTrain = np.delete(zerosTrain, to_pop, axis=0)
        # y_train = np.delete(y_train, to_pop, axis=0)
        #
        # with open('d_errs_sum', 'wb') as f:
        #     pickle.dump(d_errs_sum, f)

    return merge_histories(histories)

def prep_genos(genos, repr='f1'):
    return [gen[1:-1] for gen in genos]

def get_dissim(framsCLI, genos):
    # print(genos)
    # if checkValid:
    #     for geno in prep_genos(genos):
    #         print(framsCLI.isValid(geno))
    dissim = framsCLI.dissimilarity(prep_genos(genos))
    return dissim





def gradient(model, x):
    x_tensor = tf.convert_to_tensor(x, dtype=tf.float32)
    with tf.GradientTape() as t:
        t.watch(x_tensor)
        loss = model(x_tensor)
    return t.gradient(loss, x_tensor).numpy()


def fit_epoch(model, data, config, valid_data, framsCLI):

    X_test, zerosTest, genosTest, y_test= valid_data

    data = list(zip(*data))
    random.shuffle(data)
    batch_size = config['batch_size']
    histories = []
    # d_errs = Counter()
    for i in range(0, config['genos_per_epoch']-batch_size, batch_size):
        print("Batch, indices: ", i, " - ", i+batch_size)
        batch = data[i:i+batch_size]
        X_train, zerosTrain, genos, y_train = zip(*batch)

        try:
            dissim = get_dissim(framsCLI, genos)
        except:
            if config['locality_type'] == 'dissim':
                framsCLI = createCLI(framsCLI)
            else:
                raise BrokenPipeError("Dist matrix cannot be calculated.")
            # for gen in genos:
            #     d_errs[gen] += 1
            continue


        dissim = np.repeat(dissim[np.newaxis, :, :], batch_size, axis=0)
        # in1 = tf.Variable(np.array(X_train))
        # in2 = tf.Variable(np.array(zerosTrain))
        # in3 = tf.Variable(dissim)
        # outp = tf.constant(np.array(y_train))
        # with tf.GradientTape() as gtape:
        #     gtape.watch(model.trainable_weights)
        #     gtape.watch(model.outputs)
        #     preds = model({
        #     "inputs1": in1,
        #     "inputs2": in2
        #     ,"inputs3": in3
        # },  training=False)
        #     loss = tf.metrics.categorical_crossentropy(outp, preds)
        # grads = gtape.gradient(loss, model.trainable_weights)
        # grads2 = K.sqrt(sum([K.sum(K.square(g)) for g in grads]))
        # print("Grad norm: ", grads2.numpy())
        history = model.fit( {
            "inputs1": np.array(X_train),
            "inputs2": np.array(zerosTrain)
            ,"inputs3": tf.constant(dissim)
        }
            , np.array(y_train),
                            epochs=1, #config['epochs_per_i'],
                            verbose=2,
                            # validation_data=([X_test, zerosTest, tf.constant(genosTest)], y_test),
                            shuffle=False, batch_size=batch_size)
        histories.append(history.history)

    return merge_histories(histories)#, d_errs




