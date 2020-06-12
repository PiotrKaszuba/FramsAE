from Code.Train.prepareTrain import prepareTrain
from Code.Preparation.configuration import *



if __name__ == "__main__":
    # model_name = sys.argv[1] if sys.argv[1] != 'None' else None
    # representation = sys.argv[2] if sys.argv[2] != 'None' else None
    # long_genos = sys.argv[3] if sys.argv[3] != 'None' else None
    # cells = int(sys.argv[4]) if sys.argv[4] != 'None' else None
    # twoLayer = sys.argv[5] if sys.argv[5] != 'None' else None
    # bidir = sys.argv[6] if sys.argv[6] != 'None' else None
    # data_path = sys.argv[7] if sys.argv[7] != 'None' else None
    # load_dir = sys.argv[8] if sys.argv[8] != 'None' else None
    # onlyEval = sys.argv[9] if sys.argv[9] != 'None' else None
    model_name = 'nn'
    representation = 'f1'
    long_genos = None
    cells = 16
    twoLayer = 'oneLayer'
    bidir = 'Singledir'
    data_path = 'mods/'
    load_dir = 'newGenos/'
    onlyEval = 'False'

    config = get_config(model_name, representation, long_genos, cells, twoLayer, bidir, data_path, load_dir, onlyEval=onlyEval)
    prepareTrain(config)