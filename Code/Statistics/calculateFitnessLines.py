from textdistance import levenshtein
from typing import List, Tuple, Dict
import numpy as np
from Code.Preparation.Run_preparation_slurm import EvolutionModel
from Code.FramsticksCli import FramsticksCLI, mkdir_p
import os
import pickle
from collections import Counter

from Code.Preparation.Utils import extractFitnesses, prepareGenos
from Code.Preparation.locality_term_prep import createCLI, prep_genos
from collections import defaultdict
from scipy.spatial.distance import pdist
import matplotlib.pyplot as plt
def calculateFitnessLine(em : EvolutionModel, framsCLI: FramsticksCLI, num_points : int, magnitude: float):
    zero_point = em.means
    #direction = np.random.multivariate_normal(em.means_mut, em.cov * magnitude, size=1)[0]
    direction = (np.random.rand(em.config['cells']*2)-0.5)*2
    direction /= np.linalg.norm(direction, ord=2)
    direction *= magnitude

    all_points = np.array([zero_point+direction*(i-num_points) for i in range(2*num_points)])

    frams = em.decode_latents(all_points)
    # print(frams)
    validity = framsCLI.isValid([em.config['prefix'] + fram for fram in frams], '')
    frams_inds = [(fram, ind) for fram, valid, ind in zip(frams, validity, range(len(frams))) if valid]
    validLen = len(frams)
    print("Valid points: " + str(validLen))
    frams = [fram for fram, ind in frams_inds]
    inds = [ind for fram, ind in frams_inds]
    eval_result = framsCLI.evaluate(tuple(prep_genos(frams, em.config['representation'])))
    fitness, bad_mutants_temp, good_inds = extractFitnesses(eval_result, list(range(len(frams))))
    frams = [fram for i, fram in enumerate(frams) if i in good_inds]
    inds = [ind for i, ind in enumerate(inds) if i in good_inds]
    frams_fitn = list(zip(frams, fitness))
    # print(frams_fitn)
    return inds, fitness






def runCalculateFitnessLines(config):
    EM = EvolutionModel(config)
    FramsticksCLI.DETERMINISTIC = False  # must be set before FramsticksCLI() constructor call
    representation = config['representation'][1:]

    # importSim = 'Simulator.import(\"generation_params.sim\");' if latent == 'nolatent' else None

    framsCLI=createCLI(config)
    # framsCLI = FramsticksCLI(config['frams_path'], None, pid=config['model_name'])#, importSim=importSim)
    results = []

    for i in range(config['calculateFitnessLines_numLines']):
        print(i)
        res = calculateFitnessLine(EM, framsCLI, config['calculateFitnessLines_numPoints'], config['calculateFitnessLines_magnitude'])
        # print(res)
        results.append(res)


    # for powerr in config['calculateMutDistPowers']:
    #     if 'calculateOriginalRepresentation' not in config or not config['calculateOriginalRepresentation']:
    #
    #         res = calculateMutateDistances(config['calculateNumCentroids'], config['calculateNumMutants'], powerr, 'ldf', EM, 'latent', framsCLI)
    #     else:
    #         res = calculateMutateDistancesF(config['calculateNumCentroids'], config['calculateNumMutants'], 'ldf', powerr, EM, framsCLI)
    #
    #     print(res)
    #     results.append(res)
        save_to_shared_directory2(config, results, 'calculateFitness' if 'calculateFitnessName' not in config else config['calculateFitnessName'])

    # for rest in results:
    #     plt.plot(rest[0], rest[1])
    # plt.show()

def save_to_shared_directory2(config, result, name):
    data_path = config['data_path']
    data_dir, model_dir = os.path.split(data_path)
    shared_path = os.path.join(data_dir, "mutateDistance", model_dir)
    mkdir_p(shared_path)
    shared_path_file = os.path.join(shared_path, name)
    with open(shared_path_file, 'wb') as f:
        pickle.dump(result, f)