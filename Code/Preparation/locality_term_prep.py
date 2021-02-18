import tensorflow as tf
from Code.FramsticksCli import FramsticksCLI
import numpy as np
from collections import defaultdict
from scipy.spatial.distance import pdist, squareform
from Code.Preparation.Utils import extractFitnesses, DatasetProvider


from Code.Preparation.Run_preparation_slurm import EvolutionModel, CLIHandler, evaluate_with_batches2, CLIOrchestra
from concurrent.futures import ThreadPoolExecutor
from textdistance import levenshtein
custom_levens = lambda x,y : levenshtein(x[0], y[0])



def get_genos_dist(genos):
    XA = np.reshape(genos, (-1,1))
    return squareform(pdist(XA, custom_levens))

def dissimilarity_lev(genos):
    return get_genos_dist(genos)

def dissimilarity_rand(genos):
    return np.random.rand(len(genos), len(genos))

class LevenshteinDissim:

    def dissimilarity(self, genos):
        return get_genos_dist(genos)

class NoLocDissim:

    def dissimilarity(self, genos):
        return np.random.rand(len(genos), len(genos))


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
        if oldCLI is not None:
            importSim = oldCLI.importSim
            markers = oldCLI.markers
        else:
            importSim = config['importSim']
            markers = config['markers']
        if oldCLI is not None and level < 10:
            oldCLI.closeFramsticksCLI()
        framsCLId = FramsticksCLI(config['frams_path'], None if 'framsexe' not in config else config['framsexe'], pid=config['model_name']+str(level) + ('' if 'pid_extension' not in config else config['pid_extension']),
                                  importSim=importSim, markers=markers, config=config)
        print("OK_CORRECTLY CREATED__", level)
    except:
        return createCLI(config, oldCLI, level= level+1)
    return framsCLId

def prepareDissimGetterObject(em : EvolutionModel):
    if em.config['locality_type'] == 'dissim' or em.config['locality_type'] == 'fitness':
        em.createCLI()
        dissimGetter = em.framsCLI
    elif em.config['locality_type'] == 'levens':
        dissimGetter = LevenshteinDissim()
    else:
        dissimGetter = NoLocDissim()
    return dissimGetter

def train_with_locality(em : EvolutionModel, dp : DatasetProvider, orchestra : CLIOrchestra= None, prepared_batches=None):

    dissimGetter = prepareDissimGetterObject(em)

    histories = []
    # d_errs_sum = Counter()

    # if os.path.exists('d_errs_sum'):
    #     with open('d_errs_sum', 'rb') as f:
    #         d_errs_sum = pickle.load(f)


    for ep in range(em.config['epochs_per_i']):
        if em.config['locality_type'] == 'dissim' or em.config['locality_type'] == 'fitness':
            history = prepare_epoch(em, dp, orchestra, prepared_batches)
        else:
            history = fit_epoch(em, dp, dissimGetter, epp=ep)

        # print(em.fitnessDict)
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
import re
def prep_genos(genos, repr='f1'):
    prefix=''
    if repr == 'f9':
        prefix = '/*9*/'
    if repr == 'f4':
        prefix = '/*4*/'

    return [prefix + re.sub('[ST]', '', gen) if not gen.startswith(prefix) else re.sub('[ST]', '', gen) for gen in genos]
    # return [prefix+gen[1:-1] for gen in genos]

def get_dissim(framsCLI, genos, config, i=-1):
    # print(genos)
    # if checkValid:
    #     for geno in prep_genos(genos):
    #         print(framsCLI.isValid(geno))

    if config['locality_type'] == 'fitness':
        # eval_result = framsCLI.evaluate(tuple())
        em : EvolutionModel = framsCLI.em
        fitness, bad_mutants_temp, good_inds = em.fitness_check(prep_genos(genos, config['representation']), True, raise_err=True)
        if bad_mutants_temp > 0:
            raise Exception("Invalid fitness for at least 1 genotype")
        dissim = squareform(pdist(np.reshape(fitness, (-1,1)), 'euclidean'))
    # if config['locality_type'] == 'dissim':
    else:
        dissim = framsCLI.dissimilarity(prep_genos(genos, config['representation']))
    # if i == 0:
    #     print(dissim)
    return dissim

def get_dissim2(framsCLIHandler : CLIHandler, genos=None, fitnesses=None):
    # print(genos)
    # if checkValid:
    #     for geno in prep_genos(genos):
    #         print(framsCLI.isValid(geno))

    if framsCLIHandler.config['locality_type'] == 'fitness':
        # eval_result = framsCLI.evaluate(tuple())
        if fitnesses is not None:
            fitness = fitnesses
        else:
            if genos is None:
                raise Exception("No genos!")
            prepared_genos = prep_genos(genos, framsCLIHandler.config['representation'])
            fitness, good_inds, bad_mutants_temp = evaluate_with_batches2(framsCLIHandler, prepared_genos, base_fitness=False, return_inds=True, return_bad=True, eval_single=False)


            if bad_mutants_temp > 0:
                raise Exception("Invalid fitness for at least 1 genotype")
        dissim = squareform(pdist(np.reshape(fitness, (-1,1)), 'euclidean'))
    # if config['locality_type'] == 'dissim':
    else:
        if genos is None:
            raise Exception("No genos!")
        prepared_genos = prep_genos(genos, framsCLIHandler.config['representation'])
        dissim = framsCLIHandler.dissimilarity(prepared_genos)
    # if i == 0:
    #     print(dissim)
    return dissim





def gradient(model, x):
    x_tensor = tf.convert_to_tensor(x, dtype=tf.float32)
    with tf.GradientTape() as t:
        t.watch(x_tensor)
        loss = model(x_tensor)
    return t.gradient(loss, x_tensor).numpy()

from threading import Lock
from time import sleep
def clihandler_runf(clihandler : CLIHandler, func, lock : Lock):
    try:
        ret = func(clihandler)
        # print(ret)
        return ret
    except:
        return None
    finally:
        lock.acquire()
        clihandler.busy = False
        lock.release()

from functools import partial
def without(d, key):
    new_d = d.copy()
    new_d.pop(key)
    return new_d
def prepare_batches(em : EvolutionModel, dp : DatasetProvider, orchestra :CLIOrchestra):
    if not orchestra.initialized:
        orchestra.create_orchestra()
    data, _ = dp.get_data(em.config['genos_per_epoch'] + em.config['additional_genos_per_epoch'])
    len_all = em.config['genos_per_epoch'] + em.config['additional_genos_per_epoch']

    ask_for_objects_f_data = dp.get_ask_for_objects_f(data, dp.get_train_objects)

    batch_size = em.config['batch_size']

    if em.config['locality_type'] == 'fitness':
        eval_batch_size = batch_size * 10
        geno_batches = []
        for i in range(0, em.config['genos_per_epoch'] + em.config['additional_genos_per_epoch'] - eval_batch_size,
                       eval_batch_size):
            genos_batch = ask_for_objects_f_data('genos', i, i + eval_batch_size)
            genos_batch = [geno for geno in genos_batch if em.fitnessDict.get(geno) is None]
            geno_batches.append(genos_batch)

        f = lambda cli_handler, **kwargs: evaluate_with_batches2(cli_handler,
                                                    frams=prep_genos(kwargs['genos'],em.config['representation']),
                                                                 **without(kwargs, 'genos'))
        args_dict = dict(base_fitness=None, eval_single=False,
                      max_at_once=eval_batch_size)
        results = orchestra.run_batch(f, [{'genos': geno_batch} for geno_batch in geno_batches], args_dict)

        for res, genos_batch in zip(results, geno_batches):
            fitnesses = res.result()
            if fitnesses is not None:
                for geno, fitness in zip(genos_batch, fitnesses):
                    em.fitnessDict[geno] = fitness

        genos_all = ask_for_objects_f_data('genos')

        inds = [i for i in range(len(genos_all)) if em.fitnessDict.get(genos_all[i]) is not None]
        len_all = len(inds)
        all_data = ask_for_objects_f_data('all', indices=inds)
        ask_for_objects_f_data = dp.get_ask_for_objects_f(all_data, dp.get_train_objects)

    batches = []
    batch_datas = []
    run_args = []
    for i in range(0, len_all - batch_size, batch_size):

        batch = ask_for_objects_f_data('all', i, i + batch_size)
        ask_for_objects_f = dp.get_ask_for_objects_f(batch, dp.get_train_objects)
        genos = ask_for_objects_f('genos')
        X_train, zerosTrain = ask_for_objects_f('input')
        y_train = ask_for_objects_f('output')
        fitnesses = None
        if em.config['locality_type'] == 'fitness':
            fitnesses = [em.fitnessDict[geno] for geno in genos]

        batch_datas.append((X_train, zerosTrain, y_train))
        run_args.append({'genos':genos, 'fitnesses':fitnesses})


    results = orchestra.run_batch(get_dissim2, run_args, {})


    for res, batch_data in zip(results, batch_datas):
        dissim = res.result()
        if dissim is not None:
            batches.append((batch_data, dissim,))
    print("generated batches:", len(batches))
    return batches

def prepare_epoch(em : EvolutionModel, dp : DatasetProvider, orchestra : CLIOrchestra, prepared_batches):
    while len(prepared_batches) < em.config['steps_per_epoch']:
        prepared_batches.extend(prepare_batches(em, dp, orchestra))
    batches = [prepared_batches.pop() for i in range(em.config['steps_per_epoch'])]
    # prepared_batches = prepared_batches[em.config['steps_per_epoch']:]
    print("learning on batches: ", len(batches), ", batches left: ", len(prepared_batches))
    histories = []

    for batch_data, dissim in batches:


        dissim = np.repeat(dissim[np.newaxis, :, :], em.config['batch_size'], axis=0)
        X_train, zerosTrain, y_train = batch_data
        history = em.model.fit({
            "inputs1": np.array(X_train),
            "inputs2": np.array(zerosTrain)
            , "inputs3": tf.constant(dissim)
        }
            , np.array(y_train),
            epochs=1,  # config['epochs_per_i'],
            verbose=em.config['train_verbose'],
            # validation_data=([X_test, zerosTest, tf.constant(genosTest)], y_test),
            shuffle=False, batch_size=em.config['batch_size'])
        histories.append(history.history)

    return merge_histories(histories)  # , d_errs



def fit_epoch(em : EvolutionModel, dp : DatasetProvider, framsCLI : FramsticksCLI, epp=-1):

    # X_test, zerosTest, genosTest, y_test= valid_data
    data, _ = dp.get_data(em.config['genos_per_epoch'])
    ask_for_objects_f_data = dp.get_ask_for_objects_f(data, dp.get_train_objects)
    batch_size = em.config['batch_size']
    histories = []
    # d_errs = Counter()
    for i in range(0, em.config['genos_per_epoch']-batch_size, batch_size):
        print("Batch, indices: ", i, " - ", i+batch_size)
        batch = ask_for_objects_f_data('all', i, i+batch_size)
        ask_for_objects_f = dp.get_ask_for_objects_f(batch, dp.get_train_objects)
        # X_train, zerosTrain, genos, y_train = zip(*batch)

        try:
            dissim = get_dissim(framsCLI, ask_for_objects_f('genos'), em.config, i=epp)
        except Exception as e:
            if em.config['locality_type'] == 'dissim' or em.config['locality_type'] =='fitness':
                print("Error occured: "+str(e))
                em.createCLI()
                framsCLI = em.framsCLI
            else:
                raise BrokenPipeError("Dist matrix cannot be calculated.")
            # for gen in genos:
            #     d_errs[gen] += 1
            continue

        X_train, zerosTrain = ask_for_objects_f('input')
        y_train = ask_for_objects_f('output')
        dissim = np.repeat(dissim[np.newaxis, :, :], batch_size, axis=0)
        # in1 = tf.Variable(np.array(X_train))
        # in2 = tf.Variable(np.array(zerosTrain))
        # in3 = tf.Variable(dissim)
        # outp = tf.constant(np.array(y_train))
        # with tf.GradientTape(persistent=True) as gtape:
        #     gtape.watch(em.model.trainable_weights)
        #     gtape.watch(em.model.outputs)
        #     preds = model({
        #     "inputs1": in1,
        #     "inputs2": in2
        #     ,"inputs3": in3
        # },  training=False)
        #     loss = tf.metrics.categorical_crossentropy(outp, preds)
        # grads = gtape.gradient(loss, em.model.trainable_weights)
        # grads2 = K.sqrt(sum([K.sum(K.square(g)) for g in grads]))
        # print("Grad norm: ", grads2.numpy())


        # enc_out = em.encoder.predict({
        #     "inputs1": np.array(X_train),
        #     "inputs2": np.array(zerosTrain)
        #     ,"inputs3": tf.constant(dissim)
        # })
        # print(dissim)
        history = em.model.fit( {
            "inputs1": np.array(X_train),
            "inputs2": np.array(zerosTrain)
            ,"inputs3": tf.constant(dissim)
        }
            , np.array(y_train),
                            epochs=1, #config['epochs_per_i'],
                            verbose=em.config['train_verbose'],
                            # validation_data=([X_test, zerosTest, tf.constant(genosTest)], y_test),
                            shuffle=False, batch_size=batch_size)
        histories.append(history.history)

    return merge_histories(histories)#, d_errs




