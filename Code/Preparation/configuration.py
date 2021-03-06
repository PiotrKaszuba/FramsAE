from Code.Preparation.encodeGenos import prepareEncoders

def get_config(model_name, representation, long_genos, cells, twoLayer, bidir, data_path, load_dir, onlyEval=None, max_len=50, encoders=None, locality='0', test='0', frams_path='', clear_files = False):

    config = {}
    config['model_name'] = model_name
    config['representation'] = representation
    config['long_genos'] = long_genos if long_genos is not None else ''
    config['cells'] = cells
    config['twoLayer'] = True if twoLayer == 'twoLayer' else False
    config['bidir'] = True if bidir == 'Bidir' else False
    config['data_path'] = data_path
    config['load_dir'] = load_dir
    config['onlyEval'] = onlyEval

    config['max_len'] = max_len
    loc_type = locality[-1]
    if loc_type == 'l':
        locality = locality[:-1]
        config['locality_type'] = 'levens'
    elif loc_type == 'n':
        locality = locality[:-1]
        config['locality_type'] = 'noloc'
    else:
        config['locality_type'] = 'dissim'

    config['locality_term'] = True if locality != '0' else False
    config['locality_power'] = float(str(locality).replace('-', '.'))

    config['test'] = int(test)
    config['frams_path'] = frams_path

    if representation == 'f4':
        representation_match_part = r'/\*4\*/'
        dimsEmbed = 7
        dict = 'cfMmS>XIrFQ,iL<qTRCl '
        howMany = 507
    if representation == 'f1':
        representation_match_part = r''
        dimsEmbed = 7
        dict = 'XqQrRlLcCfFmM()iI,ST '
        howMany = 931
    if representation == 'f9':
        representation_match_part = r'/\*9\*/'
        dimsEmbed = 5
        dict = 'SRUFTDBL '
        howMany = 651

    config['representation_match_part'] = representation_match_part
    config['dimsEmbed'] = dimsEmbed
    config['dict'] = dict
    config['howMany'] = howMany

    config['features'] = len(dict)

    config['lr'] = 0.00015
    config['batch_size'] = 50
    if not config['locality_term']:
        config['batch_size'] = 1024
        config['lr'] = 0.0015

    config['reg_base'] = 2e-6

    config['past_epochs'] = 0
    config['epochs'] = 10000
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

    return config