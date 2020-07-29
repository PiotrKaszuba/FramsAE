import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

from Code.Train.prepareTrain import prepareTrain
from Code.Preparation.configuration import *
import sys


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
    # locality = sys.argv[10] if sys.argv[10] != 'None' else None
    # test = sys.argv[11] if sys.argv[11] != 'None' else None
    # frams_path = sys.argv[12] if sys.argv[12] != 'None' else None
    model_name = 'nn64LocalityBidirG'
    representation = 'f1'
    long_genos = None
    cells = 64
    twoLayer = 'oneLayer'
    bidir = 'Bidir'
    data_path = 'mods/'
    load_dir = 'newGenos/'
    onlyEval = 'False'
    locality = '0-0n'
    test = '0'
    frams_path = 'C:/Users/Piotr/Desktop/Framsticks50rc14'

    config = get_config(model_name, representation, long_genos, cells, twoLayer, bidir, data_path, load_dir, onlyEval=onlyEval, locality=locality, test=test, frams_path=frams_path)
    prepareTrain(config)