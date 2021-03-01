import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

from Code.Train.prepareTrain import prepareTrain, trainFitnessModel
from Code.Preparation.configuration import *
from Code.Evolution.frams_evolution_latent_slurm import runEvolLatent




tournsize = 3
gene = 150
pop = 50
task_test = 0




# config = get_config(model_name, representation, long_genos, cells, twoLayer, bidir, data_path, load_dir, onlyEval=onlyEval, locality=locality, test=test, frams_path=frams_path, clear_files=clear_files)
    #
    # if collect_data_plot == 'True':
    #     config['max_len'] = 100
    #     from Code.Statistics.calculateMutateDistance import runCalculate
    #     runCalculate(config)
    # else:
    #     prepareTrain(config)

from concurrent.futures import ProcessPoolExecutor
from Code.Statistics.calculateMutateDistance import runCalculate
from Code.Statistics.calculateFitnessLines import runCalculateFitnessLines
import random
from Code.FramsticksCli import fitness_min_value, fake_fitness, fake_mutate, fitness_len_weight, fitness_len_max_value, fitness_max_len, fitness_len_chars_sub

def runProcess_evol(run, locality, test, fake_f=False, latent = 'latent', evol_use_encoder=False, fake_m=False, redir_out=True, **kwargs):
    fake_fitness[0] = fake_f
    fitness_min_value[0] = -1
    fake_mutate[0] = fake_m
    fitness_len_max_value[0] = 0.0
    fitness_len_weight[0] = 0.0
    fitness_max_len[0] = 20
    # fitness_len_chars_sub[0] = '[^X]'
    model_name = 'model_%s_%s_%s_%s_%s_%s_%s' % (representation, long_genos, cells, twoLayer, bidir, locality, test)

    config = get_config(model_name, representation, long_genos, cells, twoLayer, bidir, data_path, load_dir,
                        task=task, locality=locality, test=test, frams_path=frams_path, clear_files=clear_files, tournsize=tournsize,
                        task_test=run, model_kwargs=kwargs)
    config['pid_extension'] = str(run) + 'FF'
    config['framsexe'] = 'frams.exe'
    config['mut_magnitude'] = 0.25
    config['evol_use_encoder'] = evol_use_encoder
    config['evol_keepbest'] = True
    config['cmaes'] = False
    config['onePlus'] = False
    # config['dataProvider'] = DequeMultipleCallDatasetProvider
    # config['train_data_get_methods_maker'] = get_train_set_func_f1_stream_make
    # config['test_data_get_methods_maker'] = get_test_set_func_f1_stream_make
    # config['max_len'] = 22
    # config['max_f1_len'] = 20

    runEvolLatent(config, gene=gene, pop_s=pop, latent=latent, cmaes=config['cmaes'], iterations=1, redir_out=redir_out)

def runProcess_mutDist(run, locality, test, fake_f=False, latent='latent', **kwargs):
    fake_fitness[0] = fake_f
    model_name = 'model_%s_%s_%s_%s_%s_%s_%s' % (representation, long_genos, cells, twoLayer, bidir, locality, test)

    config = get_config(model_name, representation, long_genos, cells, twoLayer, bidir, data_path, load_dir,
                        task=task, locality=locality, test=test, frams_path=frams_path, clear_files=clear_files, model_kwargs=kwargs)
    config['pid_extension'] = str(run)+'FF'

    if latent == 'latent':
        config['calculateMutDistPowers'] = [0.025, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]  # , 0.75, 1.0]
        config['calculateOriginalRepresentation'] = False
        config['mutDistName'] = model_name + '_mutDist_' + str(run)

    else:
        config['calculateMutDistPowers'] = [4, 6, 8, 10]
        config['calculateOriginalRepresentation'] = True
        config['mutDistName'] = representation + '_mutDist_' + str(run)

    # config['calculateMutDistPowers'] = [0.025, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]#, 0.75, 1.0]
    config['initial_sample_power'] = 2.0
    # random.shuffle(config['calculateMutDistPowers'])
    config['framsexe'] = 'frams-vs.exe'
    # config['mutDistName'] = model_name+'_mutDist_'+str(run)
    # config['mutDistName'] = 'f1_mutDist_' + str(run)
    config['calculateNumCentroids'] = 50
    config['calculateNumMutants'] = 50
    # config['calculateOriginalRepresentation'] = False
    config['pid_extension'] = str(run) + 'mutDist'
    # config['dataProvider'] = DequeMultipleCallDatasetProvider
    # config['train_data_get_methods_maker'] = get_train_set_func_f1_stream_make
    # config['test_data_get_methods_maker'] = get_test_set_func_f1_stream_make
    # config['max_len'] = 22

    runCalculate(config)

def runProcess_mutDist2(run, locality, test, fake_f=False):
    fake_fitness[0] = fake_f
    model_name = 'model_%s_%s_%s_%s_%s_%s_%s' % (representation, long_genos, cells, twoLayer, bidir, locality, test)

    config = get_config(model_name, representation, long_genos, cells, twoLayer, bidir, data_path, load_dir,
                        task=task, locality=locality, test=test, frams_path=frams_path, clear_files=clear_files)
    config['pid_extension'] = str(run)
    config['calculateMutDistPowers'] = [1,2,3,4,6,8,10]
    config['framsexe'] = 'frams-vs.exe'
    config['mutDistName'] = model_name+'_mutDist_'+str(run)
    # config['mutDistName'] = 'f9_mutDist_' + str(run)
    config['calculateNumCentroids'] = 20
    config['calculateNumMutants'] = 100
    config['calculateOriginalRepresentation'] = False

    runCalculate(config)


# def runProcess_train_f1(run, locality, test, fake_f=False, **kwargs):
#     fake_fitness[0] = fake_f
#     # representation = 'f1'
#     config = {}
#     model_name = 'model_%s_%s_%s_%s_%s_%s_%s' % (representation, long_genos, cells, twoLayer, bidir, locality, test)
#
#     config = get_config(model_name, representation, long_genos, cells, twoLayer, bidir, data_path, load_dir,
#                         task=task, locality=locality, test=test, frams_path=frams_path, clear_files=clear_files, model_kwargs=kwargs, existing_config=config)
#     config['pid_extension'] = str(run)
#     config['framsexe'] = 'frams-vs.exe'
#
#
#     # config['dict'] = 'XrRlL(),ST'  # F M I / maybe Q C to delete
#     # config['features'] = len(config['dict'])
#     # config['encoders'] = prepareEncoders(config['dict'])
#
#     prepareTrain(config)

def runProcess_train(run, locality, test, fake_f=False, **kwargs):
    print(bidir, twoLayer, add_params_dict)
    # return
    # random.seed(1)
    # np.random.seed(2)
    fake_fitness[0] = fake_f
    config = {}

    model_name = 'model_%s_%s_%s_%s_%s_%s_%s' % (representation, long_genos, cells, twoLayer, bidir, locality, test)

    config = get_config(model_name, representation, long_genos, cells, twoLayer, bidir, data_path, load_dir,
                        task=task, locality=locality, test=test, frams_path=frams_path, clear_files=clear_files,
                        model_kwargs=kwargs, existing_config=config)

    config['pid_extension'] = str(run)
    config['framsexe'] = 'frams-vs.exe'

    prepareTrain(config)


def runProcess_trainFitnessModel(run, locality, test, fake_f=False):
    fake_fitness[0] = fake_f
    model_name = 'model_%s_%s_%s_%s_%s_%s_%s' % (representation, long_genos, cells, twoLayer, bidir, locality, test)
    config = get_config(model_name, representation, long_genos, cells, twoLayer, bidir, data_path, load_dir,
                        task=task, locality=locality, test=test, frams_path=frams_path, clear_files=clear_files)
    config['pid_extension'] = str(run)
    config['framsexe'] = 'frams-vs.exe'
    config['epochs'] = 400

    config['dataProvider'] = DequeMultipleCallDatasetProvider
    config['train_data_get_methods_maker'] = get_train_set_func_f9_stream_make
    config['test_data_get_methods_maker'] = get_test_set_func_f9_stream_make
    config['max_len'] = 12

    trainFitnessModel(config)

def runProcess_calculateFitnessLines(run, locality, test, fake_f=False):
    fake_fitness[0] = fake_f
    model_name = 'model_%s_%s_%s_%s_%s_%s_%s' % (representation, long_genos, cells, twoLayer, bidir, locality, test)
    config = get_config(model_name, representation, long_genos, cells, twoLayer, bidir, data_path, load_dir,
                        task=task, locality=locality, test=test, frams_path=frams_path, clear_files=clear_files)
    config['pid_extension'] = str(run)
    config['framsexe'] = 'frams-vs.exe'
    config['calculateFitnessName'] = model_name + '_calculateFitness_' + str(run)
    config['calculateFitnessLines_numLines'] = 5000
    config['calculateFitnessLines_numPoints'] = 100
    config['calculateFitnessLines_magnitude'] = 0.5
    runCalculateFitnessLines(config)


# runProcess_evol(42, '3f', '2', True)
# runProcess_train(0, '3f')
# # runProcess_calculateFitnessLines(0, '0-0n')
# runProcess_mutDist2(34225, '3f', '2', False)
# runProcess_evol( 56536, '3f', '1', False, evol_use_encoder=True)
# runProcess_evol( 17 * 6 * 13 * 100, '0-0n', '0', True, 'nolatent', False, True)
number = 0
add_params_dict = {}

long_genos = None
representation = 'f9'

data_path = 'mods2/'
load_dir = 'testGenos/'

# locality = '3' # '0-0n'
frams_path = 'C:/Users/Piotr/Desktop/Framsticks50rc17'
clear_files = 'True'

if False:
    number = 8

    if number == 5:
        add_params_dict['outAdditionalDense'] = True
        add_params_dict['decoderInpDenses'] = True
    if number == 6:
        add_params_dict['outAdditionalDense'] = True
        add_params_dict['timeWindowsConstant'] = False
    if number == 7:
        add_params_dict['outAdditionalDense'] = True
        add_params_dict['inpCZeros'] = True
    if number == 8:
        add_params_dict['outAdditionalDense'] = True
        add_params_dict['inpHZeros'] = True
    if number == 9:
        add_params_dict['outAdditionalDense'] = True

#
#
#
# runProcess_evol(113*17*100, '0-0n', 0, False, 'nolatent', False, redir_out=True, **add_params_dict)
# runProcess_trainFitnessModel(4523, '3f', test=1, fake_f=False)
# all: added out dense layer
# 9 - decoder input times are ones
# 8 - same as 9 and only cell inputs / hiddens are zeros
# 7 - as 8 but cells are zeros and hiddens are inputs
# 6 - only all: ; not 9 trait
# 5 - same as 9 but also inputs to decoders pass 3 dense layers

# 100 - normal - exponential distribution learning
# 109 - exponential + 9
# 108 - exponential + 8
task = 'train'
# random.seed(73) # 20 for test, 1 for train,
# np.random.seed(98) # 30 for test, 2 for train,
cells = 64
add_params_dict['timeWindowsConstant'] = False
add_params_dict['inpHZeros'] = not add_params_dict['timeWindowsConstant']
test = '18'

# for twoLayer in ('oneLayer', 'twoLayer'):
#     for bidir in ('unidir', 'Bidir'):
#         for long_genos in (None, 'AddLayer'):
#             if long_genos is not None:
#                 add_params_dict['outAdditionalDense'] = True
#             else:
#                 add_params_dict['outAdditionalDense'] = False
#
#             runProcess_train(np.random.randint(1, 999999) + number, '0-0n', test, False, **add_params_dict)

twoLayer = 'twoLayer' # twoLayer
bidir = 'Bidir' # 'Bidir'
long_genos = None # 'AddLayer'
add_params_dict['outAdditionalDense'] = False
add_params_dict['regularization_base_latent'] = 2e-6

# compared at 300
# 0 - no reg , timeWindowsConst : 0.7893439699062814
# 1 - reg, timeWindowsConst : 0.8944921537586193
# 2 - no reg, : 0.8454011857066629
# 3 - reg, : 0.9491254602646761

# compared at 500
# 4 - reg only params : 0.9687241124151329
# 5 - reg only activation : 0.9593493058810697
# 6 - reg, inpHzeros
# 9 - reg only params, inpHzeros
# 10 - reg only activation, inpHzeros


# 11 - reg, inpHzeros, addLayer
# 12 - reg only activation, inpHzeros, addLayer
# 13 - reg, inpHzeros, addLayer, twoLayer
# 14 - reg only activation, inpHzeros, twoLayer
# 15 - reg only activation, inpHzeros, addLayer, twoLayer
# 16 - reg, inpHzeros, twoLayer



# runProcess_mutDist(3532, '0-0', 0)

# runProcess_evol(57*8659, '3f', test, fake_f=False, latent='latent', evol_use_encoder=False, fake_m=True, redir_out=False, **add_params_dict)
# runProcess_train(131*43, '3f', test, False, **add_params_dict)
runProcess_evol(352*943, '3f', '19', latent='nolatent', fake_m=True, redir_out=False, **add_params_dict)
# for i in [17, 19, 23]:
# runProcess_mutDist(127*43*97, '0-0n', test, False, latent='nolatent', **add_params_dict)
raise NotImplementedError()

if __name__ == '__main__':
    e = ProcessPoolExecutor(
                max_workers=1,
            )

    for twoLay in ('oneLayer', 'twoLayer'):
        twoLayer = twoLay
        for bidi in ('unidir', 'Bidir'):
            bidir = bidi
            for long_geno in (None, 'AddLayer'):
                long_genos = long_geno
                if long_genos is not None:
                    add_params_dict['outAdditionalDense'] = True
                else:
                    add_params_dict['outAdditionalDense'] = False

                f=e.submit(runProcess_train, np.random.randint(1, 999999) + number, '0-0n', test, False, **add_params_dict)
    f.result()
    raise NotImplementedError()
    for i in [131]:
        # j=i+35
        j = i

        # f=e.submit(runProcess_mutDist, j*2, '0-0n', 0)
        # e.submit(runProcess_mutDist, j * 3, '3l', 0)
        # e.submit(runProcess_mutDist, j * 4, '3', 0)
        # e.submit(runProcess_mutDist, j * 5, '3f', 0)
        # e.submit(runProcess_train, 0, '0-0n')
        # e.submit(runProcess_train_f1, 1, '3l',0)
        # e.submit(runProcess_train_f1, 2, '3f',0)
        # e.submit(runProcess_train_f1, 3, '3',0)
#         # e.submit(runProcess_calculateFitnessLines, j, '0-0n')
#         # e.submit(runProcess_calculateFitnessLines, j, '3l')
#         # e.submit(runProcess_calculateFitnessLines, j, '3f')
#         # e.submit(runProcess_calculateFitnessLines, j, '3')
#         # e.submit(runProcess_mutDist, j, '0-0n')
        for use_enc, multip in zip([False, True], [11, 13]):
            f=e.submit(runProcess_evol, j * 2 * multip, '3f', '0', False, 'latent', use_enc)
            e.submit(runProcess_evol, j * 3 * multip, '0-0n', '0', False, 'latent', use_enc)
            e.submit(runProcess_evol, j * 4 * multip, '3', '0', False, 'latent', use_enc)
            e.submit(runProcess_evol, j * 5 * multip, '3l', '0', False, 'latent', use_enc)
#
            e.submit(runProcess_evol, j * 6 * multip, '0-0n', '0', False, 'nolatent', use_enc, False)
#             #
#             # e.submit(runProcess_evol, j * 7 * multip, '0-0n', '0', True, 'nolatent', use_enc, True)
#
#             e.submit(runProcess_evol, j * 8 * multip, '3f', '2', True, 'latent', use_enc)
#             e.submit(runProcess_evol, j* 9 * multip, '0-0n', '1', True, 'latent', use_enc)
#             e.submit(runProcess_evol, j * 10 * multip, '3l', '1', True, 'latent', use_enc)
#         break
# #     # runProcess_mutDist(0, '0-0n')
# #     # runProcess_train(0, '3f')
# #
# # #
# # # # e.submit()
# # #
# # # # prepareTrain(config)
# # #
    print(f.result())