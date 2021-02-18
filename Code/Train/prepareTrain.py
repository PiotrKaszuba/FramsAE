from Code.Preparation.ParseGenFile import testGenos
from Code.Preparation.encodeGenos import prepareEncoders,encodeGenos, inverse
import tensorflow as tf
pad_sequences = tf.keras.preprocessing.sequence.pad_sequences

from Code.Preparation.locality_term_prep import train_with_locality, dissimilarity_lev, dissimilarity_rand
from Code.Preparation.Run_preparation_slurm import EvolutionModel, CLIHandler, CLIOrchestra
from Code.Preparation.Utils import DatasetProvider
import os
import numpy as np
import random


def trainFitnessModel(config):
    em = EvolutionModel(config)
    dp: DatasetProvider = config['dataProvider'](config['train_data_get_methods_maker'],
                                                           config['test_data_get_methods_maker'], config)
    em.createCLI()
    for i in range(config['epochs']):
        data, _ = dp.get_data(em.config['genos_per_epoch'])
        ask_for_objects_f_data = dp.get_ask_for_objects_f(data, dp.get_train_objects)
        batch_size = em.config['batch_size']
        histories = []
        # d_errs = Counter()
        vals_to_occur = list(range(0, em.config['genos_per_epoch'] - batch_size, batch_size))
        for i in range(0, em.config['genos_per_epoch'] - batch_size, batch_size):
            print("Batch, indices: ", i, " - ", i + batch_size)
            batch = ask_for_objects_f_data('all', i, i + batch_size)
            ask_for_objects_f = dp.get_ask_for_objects_f(batch, dp.get_train_objects)

        # _, ask_for_objects_f_train = dataProvider.get_data()
        # _, ask_for_objects_f_test = dataProvider.get_test_data()


            inputs, outputs = em.prepare_fitnessPrediction(ask_for_objects_f('input'), k=1)

            # print(inputs, outputs)
            # p = em.fitnessModel.predict
            if i == vals_to_occur[0]:
                pred = em.fitnessModel.predict(inputs, batch_size=config['batch_size'])
                res =  ((outputs-pred)**2).mean()
                # print("res", res)
                em.summarize_train_phase_fitnessModel(res)

            else:
                history = em.fitnessModel.fit(inputs, outputs,
                                       epochs=config['epochs_per_i'],
                                       verbose=2,
                                       # validation_data=(ask_for_objects_f_test('input'), ask_for_objects_f_test('output')),
                                       shuffle=True, batch_size=config['batch_size'])
            # history = history.history


def prepareTrain(config):
    # os.environ['PYTHONHASHSEED'] = '0'
    # np.random.seed(42)
    # random.seed(12345)
    # tf.random.set_seed(1)
    #model_path_base = os.path.join(config['data_path'], config['model_name'] + str(config['past_epochs'])) + "_" + str(config['loaded_model_acc'])

    em = EvolutionModel(config)
    # em.fill_geno_bank()
    dataProvider : DatasetProvider = config['dataProvider'](config['train_data_get_methods_maker'], config['test_data_get_methods_maker'], config, em=em)


    def evaluate():
        _, ask_for_objects_f = dataProvider.get_test_data()
        acc = em.evaluate(ask_for_objects_f('input'), ask_for_objects_f('genosCheck'))
        return acc

    accStart = evaluate()

    if accStart == 0.0 or config['task'] == 'evaluate_model':
        return


    if em.config['locality_type'] == 'dissim' or em.config['locality_type'] == 'fitness':
        prepared_batches = []
    else:
        prepared_batches = None

    for i in range(config['epochs']):
        if em.epochs_total >= config['epochs']:
            break
        if config['locality_term']:
            history = train_with_locality(em, dataProvider, orchestra=em.cli_orchestra, prepared_batches=prepared_batches)
        else:
            _, ask_for_objects_f_train = dataProvider.get_data()
            _, ask_for_objects_f_test = dataProvider.get_test_data()
            history = em.model.fit(ask_for_objects_f_train('input'), ask_for_objects_f_train('output'),
                                epochs=config['epochs_per_i'],
                                verbose=2,
                                validation_data=(ask_for_objects_f_test('input'), ask_for_objects_f_test('output')),
                                shuffle=True, batch_size=config['batch_size'])
            history = history.history

        acc = evaluate()
        if acc >= em.config['accuracy_target']:
            return
        if acc < 0.275 and em.epochs_total > 10:
            prepareTrain(config)
            return

        em.summarize_train_phase(acc, history, i)








