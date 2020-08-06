from textdistance import levenshtein
from typing import List
import numpy as np
from Code.Preparation.Run_preparation_slurm import EvolutionModel, prepareGenos
from Code.FramsticksCli import FramsticksCLI, mkdir_p
import os
import pickle
def distanceLevenhstein(centroid : str, mutants : List[str])->float:
    return float(np.mean([levenshtein(centroid, mutant) for mutant in mutants]))

def mutate_latent(centroid : np.ndarray, times, power, evolution_model : EvolutionModel, framsCLI : FramsticksCLI) -> List[str]:
    latent_mutants = np.array([evolution_model.mutate(centroid, 1, power) for i in range(times*2)])

    frams = evolution_model.decode_latents(latent_mutants)
    validity = framsCLI.isValid(frams, 'mutate')
    frams = [frams for frams, valid in zip(frams, validity) if valid]
    validLen = len(frams)

    if validLen>= times:
        return frams[:times]
    else:
        return frams + mutate_latent(centroid, times-validLen, power, evolution_model, framsCLI)

def calculateMutateDistance(centroid : str, times, power, distance, evolutionModel : EvolutionModel, representation='latent', framsCLI=None):

    if representation == 'latent':
        prepared_centroid = prepareGenos([centroid], evolutionModel.config)
        enc_latent = evolutionModel.predictEncoderLatent(prepared_centroid)
        centroid_latent = enc_latent[0]

        mutants = mutate_latent(centroid_latent, times, power, evolutionModel)
    else:
        raise NotImplementedError()

    if distance == 'l':
        dist = distanceLevenhstein(centroid, mutants)
    else:
        raise NotImplementedError()
    return dist

def getCentroids_latent(num_centroids, evolution_model : EvolutionModel, framsCLI : FramsticksCLI) -> List[str]:
    samples = evolution_model.sampleMultivariateGaussianLatent(num_centroids*20)

    frams = evolution_model.decode_latents(samples)
    validity = framsCLI.isValid(frams, 'centroid')
    frams = [frams for frams, valid in zip(frams, validity) if valid]
    validLen = len(frams)

    if validLen >= num_centroids:
        return frams[:num_centroids]
    else:
        return frams + getCentroids_latent(num_centroids-validLen, evolution_model, framsCLI)



def calculateMutateDistances(num_centroids, times, power, distance, evolutionModel : EvolutionModel, representation='latent', framsCLI = None):
    centroids = getCentroids_latent(num_centroids, evolutionModel)

    mut_dist = np.mean([calculateMutateDistance(centroid, times, power, distance, evolutionModel, representation, framsCLI) for centroid in centroids])
    return mut_dist

def save_to_shared_directory(config, result, name):
    data_path = config['data_path']
    data_dir, model_dir = os.path.split(data_path)
    shared_path = os.path.join(data_dir, "mutateDistance", model_dir)
    mkdir_p(shared_path)
    shared_path_file = os.path.join(shared_path, name)
    with open(shared_path_file, 'wb') as f:
        pickle.dump(result, f)



def runCalculate(config):
    EM = EvolutionModel(config)
    FramsticksCLI.DETERMINISTIC = False  # must be set before FramsticksCLI() constructor call
    representation = config['representation'][1:]

    # importSim = 'Simulator.import(\"generation_params.sim\");' if latent == 'nolatent' else None
    framsCLI = FramsticksCLI(config['frams_path'], None, pid=config['model_name'])#, importSim=importSim)
    res = calculateMutateDistances(10, 10, 0.01, 'l', EM, 'latent', framsCLI)
    print(res)
    save_to_shared_directory(config, res, 'mutDist_l')

