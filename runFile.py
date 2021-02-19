import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
from Code.Evolution.frams_evolution_latent_slurm import runEvolLatent
from Code.Train.prepareTrain import prepareTrain
from Code.Preparation.configuration import *
from Code.FramsticksCli import fake_fitness, fitness_min_value, fake_mutate, fitness_len_weight, fitness_len_max_value, fitness_max_len, fitness_len_chars_sub

import sys
if __name__ == "__main__":

    model_name = sys.argv[1] if sys.argv[1] != 'None' else None
    representation = sys.argv[2] if sys.argv[2] != 'None' else None
    long_genos = sys.argv[3] if sys.argv[3] != 'None' else None
    cells = int(sys.argv[4]) if sys.argv[4] != 'None' else None
    twoLayer = sys.argv[5] if sys.argv[5] != 'None' else None
    bidir = sys.argv[6] if sys.argv[6] != 'None' else None
    data_path = sys.argv[7] if sys.argv[7] != 'None' else None
    load_dir = sys.argv[8] if sys.argv[8] != 'None' else None
    locality = sys.argv[10] if sys.argv[10] != 'None' else None
    test = sys.argv[11] if sys.argv[11] != 'None' else None
    frams_path = sys.argv[12] if sys.argv[12] != 'None' else None
    clear_files = sys.argv[13] if sys.argv[13] != 'None' else None

    latent = sys.argv[14] if sys.argv[14] != 'None' else None
    tournsize = int(sys.argv[15]) if sys.argv[15] != 'None' else None
    gene = int(sys.argv[16]) if sys.argv[16] != 'None' else None
    pop = int(sys.argv[17]) if sys.argv[17] != 'None' else None
    task_test = int(sys.argv[18]) if sys.argv[18] != 'None' else None
    task = sys.argv[9] if sys.argv[9] != 'None' else None

    ## CONSTANT
    add_params_dict = {}
    add_params_dict['timeWindowsConstant'] = False
    add_params_dict['inpHZeros'] = not add_params_dict['timeWindowsConstant']

    add_params_dict['outAdditionalDense'] = False
    add_params_dict['regularization_base_latent'] = 2e-6

    fitness_len_max_value[0] = 0.0
    fitness_len_weight[0] = 0.0
    fitness_max_len[0] = 20
    fake_mutate[0] = True
    fitness_min_value[0] = -1 #if task_test%19 !=0 else 0.0
    # random.seed(70)  # 20 for test, 1 for train,
    # np.random.seed(80)  # 30 for test, 2 for train,

    ###################
    config = {}
    config = get_config(model_name, representation, long_genos, cells, twoLayer, bidir, data_path, load_dir, task=task,
                        locality=locality, test=test, frams_path=frams_path, clear_files=clear_files, tournsize=tournsize, task_test=task_test,
                        model_kwargs=add_params_dict, existing_config=config)

    config['pid_extension'] = str(task_test) + "_" + str(np.random.randint(0, 9999))
    print(config)

    if task == 'evol':
        # config['framsexe'] = 'frams-vs.exe'
        config['mut_magnitude'] = 0.2
        config['evol_use_encoder'] = False #if task_test % 131 !=0 else True
        config['evol_keepbest'] = True
        config['cmaes'] = False # if task_test % 7 !=0 else True

        config['let_not_newest_weights'] = True
        redir_out = True
        iterations = 1
        if not latent == 'latent':
            iterations = 4

        runEvolLatent(config, gene=gene, pop_s=pop, latent=latent, cmaes=config['cmaes'], iterations=iterations,
                      redir_out=redir_out)

    elif task == 'train':
        prepareTrain(config)
    elif task == 'collect_data':
        from Code.Statistics.calculateMutateDistance import runCalculate

        if latent == 'latent':
            config['calculateMutDistPowers'] = [0.025, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]  # , 0.75, 1.0]
            config['calculateOriginalRepresentation'] = False
            config['mutDistName'] = model_name + '_mutDist_' + str(task_test)

        else:
            config['calculateMutDistPowers'] = [1, 2, 3, 4, 6, 8, 10]
            config['calculateOriginalRepresentation'] = True
            config['mutDistName'] = representation + '_mutDist_' + str(task_test)

        config['initial_sample_power'] = 2.0
        config['calculateNumCentroids'] = 50
        config['calculateNumMutants'] = 50

        runCalculate(config)
    elif task == 'FDC':
        from Code.Statistics.gatherResultsFDC import runFDCforModel
        runFDCforModel(model_name, data_path)



    # runEvol(evol_name='test', representation='f1', data_path='mods', load_dir='',
    #         frams_path='C:/Users/Piotr/Desktop/Framsticks50rc14')

    # model_name = 'nn64LocalityBidirG'
    # representation = 'f1'
    # long_genos = None
    # cells = 64
    # twoLayer = 'oneLayer'
    # bidir = 'Bidir'
    # data_path = 'mods/'
    # load_dir = 'newGenos/'
    # onlyEval = 'False'
    # locality = '0-0n'
    # test = '0'
    # frams_path = 'C:/Users/Piotr/Desktop/Framsticks50rc14'