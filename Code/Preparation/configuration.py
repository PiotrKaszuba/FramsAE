from Code.Preparation.encodeGenos import prepareEncoders
from Code.Preparation.Utils import *




def get_config(model_name, representation, long_genos, cells, twoLayer, bidir, data_path, load_dir, task=None, max_len=50, encoders=None, locality='0', test='0', frams_path='', clear_files = False, tournsize=None, task_test=0, model_kwargs= None, existing_config=None):

    config = {} if existing_config is None else existing_config
    config['model_name'] = model_name
    config['representation'] = representation
    config['long_genos'] = long_genos if long_genos is not None else ''
    config['cells'] = cells
    config['twoLayer'] = True if twoLayer == 'twoLayer' else False
    config['bidir'] = True if bidir == 'Bidir' else False
    config['data_path'] = data_path
    config['load_dir'] = load_dir
    config['task'] = task

    config['max_len'] = max_len
    loc_type = locality[-1]
    if loc_type == 'l':
        locality = locality[:-1]
        config['locality_type'] = 'levens'
    elif loc_type == 'n':
        locality = locality[:-1]
        config['locality_type'] = 'noloc'
    elif loc_type == 'f':
        locality = locality[:-1]
        config['locality_type'] = 'fitness'
    else:
        config['locality_type'] = 'dissim'

    config['locality_term'] = True if locality != '0' else False
    config['locality_power'] = float(str(locality).replace('-', '.'))

    config['test'] = int(test)
    config['frams_path'] = frams_path

    if representation == 'f4':
        representation_match_part = r'/\*4\*/'
        dimsEmbed = 8
        dict = 'cfMmS>XIrFQ,iL<qTRCl'
        howMany = 507
        config['prefix'] = '/*4*/'
        config['check_valid'] = True
    if representation == 'f1':
        representation_match_part = r''
        dimsEmbed = 7 # 7
        dict = 'XrRlL(),ST' # F M I / maybe Q C to delete
        howMany = 931
        config['prefix'] = ''
        config['check_valid'] = True
        config['train_data_get_methods_maker'] = get_train_set_func_f1_stream_make
        config['test_data_get_methods_maker'] = get_test_set_func_f1_stream_make
        config['max_len'] = 22
        config['max_f1_len'] = 20
        distr_a = 0.25
        distr_b = -2
        distr_base = 2

    if representation == 'f9':
        representation_match_part = r'/\*9\*/'
        dimsEmbed = 5 # 5
        dict = 'SRUFTDBL'
        howMany = 651
        config['prefix'] = '/*9*/'
        config['check_valid'] = False
        config['train_data_get_methods_maker'] = get_train_set_func_f9_stream_make
        config['test_data_get_methods_maker'] = get_test_set_func_f9_stream_make
        config['max_len'] = 22
        config['max_f1_len'] = 20
        distr_a = 0.15
        distr_b = -2
        distr_base = 2
        config['generate_test_batch_multip'] = 1

    config['representation_match_part'] = representation_match_part
    config['dimsEmbed'] = dimsEmbed
    config['dict'] = dict
    config['howMany'] = howMany

    config['features'] = len(dict)
    config['min_lr'] = 0.00005
    config['lr_epochs_done_decay'] = 0.995
    config['lr'] = 0.001
    config['batch_size'] = 100
    if not config['locality_term']:
        config['batch_size'] = 1024
        config['lr'] = 0.0015

    config['reg_base'] = 2e-6

    config['past_epochs'] = 0
    # config['epochs'] = 10000
    config['epochs'] = 2001
    config['epochs_per_i'] = 1
    if not config['locality_term']:
        config['epochs_per_i'] = 20



    config['genos_per_epoch'] = 5001

    config['loaded_model_acc'] = 0

    config['oneHot'] = False
    config['reverse'] = False

    if encoders is None:
        encoders = prepareEncoders(config['dict'])

    config['encoders'] = encoders
    config['clear_files'] = True if clear_files == 'True' else False

    # config['dataProvider'] = OneCallDatasetProvider
    config['dataProvider'] = DequeMultipleCallDatasetProvider

    # config['train_data_get_methods_maker'] = get_train_set_func_make
    # config['test_data_get_methods_maker'] = get_test_set_func_make



    config['test_deque_size'] = 20000

    config['importSim'] = []
    config['markers'] = []
    config['importSim'].append('Simulator.import(\"generation_params.sim\");')
    config['markers'].append("Simulator.load")

    config['importSim'].extend(["Simulator.print(SimilMeasureHungarian.simil_partdeg=1.0);",
                                "Simulator.print(SimilMeasureHungarian.simil_partgeom=1.0);"])
    config['markers'].extend(["Script.Message", "Script.Message"])


    if representation == 'f1':
        dict_cleaned = re.sub('[ST]', '', dict)

        config['importSim'].append(
            f'for(var i=0; i<String.len("{dict_cleaned}"); i++) GenMan.f1_mut_exmod = String.replace(GenMan.f1_mut_exmod, String.substr("{dict_cleaned}", i, 1), ""); Simulator.print(GenMan.f1_mut_exmod);'
        )
        config['markers'].append("Script.Message")
    if tournsize is None:
        config['tourney_size'] = 3
    else:
        config['tourney_size'] = tournsize
    config['task_test'] = task_test
    config['evol_use_encoder'] = False
    config['evol_keepbest'] = True
    # generation of genos:
    config['generate_size'] = 20000
    config['close_range'] = 5
    config['starting_points'] = 5000
    config['starting_add_attempts'] = 5000
    config['starting_points_batch'] = 2500
    config['diversity'] = 12
    config['need_value_proximity_times'] = 1
    config['generate_timeout'] = 300
    # config['max_f1_len'] = 20

    config['model_kwargs'] = {} if model_kwargs is None else model_kwargs
    config['starting_point_len_distribution'] = make_probe_distribution_f(make_lengths(1, config['max_f1_len']), starting_points_distribution(1, config['max_f1_len'], 0.5, 1, 0, 2))
    config['geno_len_distribution'] = make_probe_distribution_f(make_lengths(1, config['max_f1_len']), exponential_distribution(1, config['max_f1_len'], distr_a, distr_b, distr_base))

    config['requested_epoch'] = None
    config['spareWeightFilesFunc'] = spareWeightFunc
    config['train_verbose'] = 2

    config['saveFilesEveryITimes'] = 1
    config['additional_genos_per_epoch'] = 500
    config['train_deque_size'] = config['genos_per_epoch'] + config['additional_genos_per_epoch']
    config['locality_prepare_workers'] = 3

    config['steps_per_epoch'] = 50
    config['let_not_newest_weights'] = True
    config['accuracy_target'] = 0.99
    return config