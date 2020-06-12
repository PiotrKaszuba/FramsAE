from Code.Preparation.encodeGenos import prepareEncoders

def get_config(model_name, representation, long_genos, cells, twoLayer, bidir, data_path, load_dir, onlyEval=None, max_len=50, encoders=None):

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

    config['lr'] = 0.0005
    config['batch_size'] = 2048
    config['reg_base'] = 2e-6

    config['past_epochs'] = 0
    config['epochs'] = 2000
    config['epochs_per_i'] = 10

    config['loaded_model_acc'] = 0

    config['oneHot'] = False
    config['reverse'] = False

    if encoders is None:
        encoders = prepareEncoders(config['dict'])

    config['encoders'] = encoders


    return config