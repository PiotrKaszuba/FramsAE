from textdistance import levenshtein
from typing import List, Tuple, Dict
import numpy as np
from Code.Preparation.Run_preparation_slurm import EvolutionModel, checkValid
from Code.FramsticksCli import FramsticksCLI, mkdir_p
import os
import pickle
from collections import Counter

from Code.Preparation.Utils import extractFitnesses, prepareGenos
from Code.Preparation.locality_term_prep import createCLI
from collections import defaultdict
from scipy.spatial.distance import pdist
from Code.FramsticksCli import fake_fitness
class BadCentroidException(Exception):
    pass

custom_levens = lambda x, y: levenshtein(x[0], y[0])

def corrLevenshtein(frams:List[str], latent_distances: np.ndarray, latent_distances_encoder:np.ndarray, suffix='_n2') -> Dict:
    XA = np.reshape(frams, (-1, 1))
    distances_levens = pdist(XA, custom_levens)

    corr = np.corrcoef(distances_levens, latent_distances)
    corr_encoder = np.corrcoef(distances_levens, latent_distances_encoder)
    min = float(np.min(distances_levens))
    max = float(np.max(distances_levens))
    avg = float(np.mean(distances_levens))
    var = float(np.var(distances_levens))
    rang = max - min

    return {'corr_l'+suffix: corr, 'corrEncoder_l'+suffix: corr_encoder, 'avg_l'+suffix: avg, 'var_l'+suffix: var, 'max_l'+suffix: max, 'min_l'+suffix: min, 'rang_l'+suffix: rang}




def distanceLevenhstein(centroid : str, mutants : List[str] , latent_mutants: np.ndarray=None, latent_mutants_encoder:np.ndarray=None)->Dict:
    lev = [levenshtein(centroid, mutant) for mutant in mutants]

    min = float(np.min(lev))
    max = float(np.max(lev))
    avg = float(np.mean(lev))
    var = float(np.var(lev))
    rang = max-min
    if latent_mutants is not None:
        latent_distances = pdist(latent_mutants, 'euclidean')
        latent_distances_encoder = pdist(latent_mutants_encoder, 'euclidean')
        corrOuts = corrLevenshtein(mutants, latent_distances, latent_distances_encoder)
    else:
        corrOuts = {}
    return {**corrOuts, **{'avg_l':avg, 'var_l':var, 'max_l':max, 'min_l':min, 'rang_l':rang}}

def corrDissim(dissims: List[np.ndarray], latent_distances: List[np.ndarray], latent_distances_encoder:List[np.ndarray], suffix='_n2'):
    total_vec_dissim = []
    total_vec_latent = []
    total_vec_latent_encoder = []
    for dissim, latent_distance, latent_distance_encoder in zip(dissims, latent_distances, latent_distances_encoder):
        inds = np.triu_indices(len(dissim), 1)
        dissim_vec = dissim[inds]
        total_vec_dissim = np.concatenate([total_vec_dissim, dissim_vec])
        total_vec_latent = np.concatenate([total_vec_latent, latent_distance])
        total_vec_latent_encoder = np.concatenate([total_vec_latent_encoder, latent_distance_encoder])
    corr = np.corrcoef(total_vec_dissim, total_vec_latent)
    corr_encoder = np.corrcoef(total_vec_dissim, total_vec_latent_encoder)
    min = float(np.min(total_vec_dissim))
    max = float(np.max(total_vec_dissim))
    avg = float(np.mean(total_vec_dissim))
    var = float(np.var(total_vec_dissim))
    rang = max - min
    return {'corr_d'+suffix: corr, 'corrEncoder_d'+suffix: corr_encoder, 'avg_d'+suffix: avg, 'var_d'+suffix: var, 'max_d'+suffix: max, 'min_d'+suffix: min, 'rang_d'+suffix: rang}


def distanceDissim(centroid : str, mutants : List[str], framsCLI : FramsticksCLI, evolution_model : EvolutionModel, latents:np.ndarray=None, latents_encoder:np.ndarray=None)->Dict:
    max_at_once = 50
    bad_mutants = 0
    dissims = []
    matrices = []
    latent_matrices = []
    latent_matrices_encoder = []
    for i in range(0, len(mutants), max_at_once):
        temp = mutants[i:i+max_at_once]
        temp = [evolution_model.config['prefix']+tem for tem in temp]
        temp.insert(0, evolution_model.config['prefix']+centroid)

        try:
            dissim = evolution_model.framsCLI.dissimilarity(temp)
            matrices.append(dissim[1:,1:])
            if latents is not None:
                latent_matrices.append(latents[i:i+max_at_once])
                latent_matrices_encoder.append(latents_encoder[i:i+max_at_once])
            dissim = dissim[0]

            bad_mutants += max_at_once - len(dissim)
        except:
            # framsCLI = createCLI(evolution_model.config, framsCLI)
            evolution_model.createCLI()
            dissim = []
            for q in range(1, len(temp)):
                t = [evolution_model.config['prefix']+centroid, temp[q]]
                try:
                    dissimPart = evolution_model.framsCLI.dissimilarity(t)
                    dissim.append(dissimPart[0, 1])
                except:
                    bad_mutants += 1
                    framsCLI = createCLI(evolution_model.config, framsCLI)
        dissims.extend(dissim)


    if latents is not None:
        latent_matrices = [pdist(l, 'euclidean') for l in latent_matrices]
        latent_matrices_encoder = [pdist(l, 'euclidean') for l in latent_matrices_encoder]
    min = float(np.min(dissims))
    max = float(np.max(dissims))
    avg = float(np.mean(dissims))
    var = float(np.var(dissims))
    rang = max - min
    corrOuts = corrDissim(matrices, latent_matrices, latent_matrices_encoder) if latents is not None else {}
    return {**corrOuts,**{'avg_d': avg, 'var_d': var, 'max_d': max, 'min_d': min, 'rang_d': rang, 'bad_mutants_d': bad_mutants}}


def corrFitness(fitnesses:List[float], latent_distances:np.ndarray, latent_distances_encoder:np.ndarray, suffix='_n2'):
    distances_fitness = pdist(np.reshape(fitnesses, (-1,1)))

    corr = np.corrcoef(distances_fitness, latent_distances)
    corr_encoder = np.corrcoef(distances_fitness, latent_distances_encoder)
    min = float(np.min(distances_fitness))
    max = float(np.max(distances_fitness))
    avg = float(np.mean(distances_fitness))
    var = float(np.var(distances_fitness))
    rang = max - min
    return {'corr_f'+suffix: corr, 'corrEncoder_f'+suffix: corr_encoder, 'avg_f'+suffix: avg, 'var_f'+suffix: var, 'max_f'+suffix: max, 'min_f'+suffix: min, 'rang_f'+suffix: rang}



def distanceFitness(centroid : str, mutants : List[str], framsCLI : FramsticksCLI, evolution_model : EvolutionModel, latents:np.ndarray=None, latents_encoder:np.ndarray=None, addFitnesses=False, mutants_decoder=None)->Dict:
    # max_at_once = 50
    # bad_mutants = 0
    # fitnesses = []
    # latent_indices = []
    centroid_eval, bad_centroid, _ = evolution_model.fitness_check([centroid], True)
    # centroid_fitness, bad_centroid, _ = extractFitnesses(centroid_eval, [0])
    if bad_centroid >0 or centroid_eval[0] is None:
        raise BadCentroidException
    centroid_fitness = centroid_eval[0]
    # for i in range(0, len(mutants), max_at_once):
    #     temp = mutants[i:i + max_at_once]
    #     temp = [evolution_model.config['prefix'] + tem for tem in temp]
    #     # temp.insert(0, centroid)
    #
    #     try:
    #         eval_result = framsCLI.evaluate(tuple(temp))
    #         fitness, bad_mutants_temp, good_inds = extractFitnesses(eval_result, list(range(i, i + max_at_once)))
    #         latent_indices.extend(good_inds)
    #         bad_mutants += bad_mutants_temp
    #     except:
    #         framsCLI = createCLI(evolution_model.config, framsCLI)
    #         fitness = []
    #         for q in range(len(temp)):
    #             t = (temp[q],)
    #             try:
    #                 fitnPart_eval = framsCLI.evaluate(t)
    #                 fitnPart, bad_mutants_temp, good_ind = extractFitnesses(fitnPart_eval, [i + q])
    #
    #                 if bad_mutants_temp > 0:
    #                     bad_mutants += 1
    #                 else:
    #                     fitness.append(fitnPart[0])
    #                     latent_indices.append(good_ind)
    #             except:
    #                 bad_mutants += 1
    #                 framsCLI = createCLI(evolution_model.config, framsCLI)
    #     fitnesses.extend(fitness)
    fitnesses, bad_mutants, latent_indices = evolution_model.fitness_check(mutants, True)
    latent_indices2 = None
    fitness2_decoder = None
    if mutants_decoder is not None:
        fitness2_decoder, _, latent_indices2 = evolution_model.fitness_check(mutants, True)

    #     latent_indices2 = []
    #     fitness2_decoder = []
    #     for i in range(0, len(mutants), max_at_once):
    #         temp = mutants_decoder[i:i + max_at_once]
    #         temp = [evolution_model.config['prefix'] + tem for tem in temp]
    #         # temp.insert(0, centroid)
    #
    #         try:
    #             eval_result = framsCLI.evaluate(tuple(temp))
    #             fitness, bad_mutants_temp, good_inds = extractFitnesses(eval_result, list(range(i, i + max_at_once)))
    #             latent_indices2.extend(good_inds)
    #             # bad_mutants += bad_mutants_temp
    #         except:
    #             framsCLI = createCLI(evolution_model.config, framsCLI)
    #             fitness = []
    #             for q in range(len(temp)):
    #                 t = (temp[q],)
    #                 try:
    #                     fitnPart_eval = framsCLI.evaluate(t)
    #                     fitnPart, bad_mutants_temp, good_ind = extractFitnesses(fitnPart_eval, [i + q])
    #
    #                     if bad_mutants_temp > 0:
    #                         bad_mutants += 1
    #                     else:
    #                         fitness.append(fitnPart[0])
    #                         latent_indices2.append(good_ind)
    #                 except:
    #                     bad_mutants += 1
    #                     framsCLI = createCLI(evolution_model.config, framsCLI)
    #         fitness2_decoder.extend(fitness)

    if latent_indices2 is not None:
        latent_indices = list(set(latent_indices).intersection(set(latent_indices2)))
        mutants_decoder_to_pick = np.array(mutants_decoder)[latent_indices]
        fitness2_decoder = list(np.array(fitness2_decoder)[latent_indices])

    fitnesses = list(np.array(fitnesses)[latent_indices])
    mutants_to_pick = np.array(mutants)[latent_indices]
    if latents is not None:
        latents_to_pick = latents[latent_indices]
        latents_to_pick_encoder = latents_encoder[latent_indices]

        latent_dists = pdist(latents_to_pick, 'euclidean')
        latent_dists_encoder = pdist(latents_to_pick_encoder, 'euclidean')

    fitnesses_diff = [abs(fitn-centroid_fitness) for fitn in fitnesses]
    min = float(np.min(fitnesses_diff))
    max = float(np.max(fitnesses_diff))
    avg = float(np.mean(fitnesses_diff))
    var = float(np.var(fitnesses_diff))
    rang = max - min

    minT = float(np.min(fitnesses))
    maxT = float(np.max(fitnesses))
    avgT = float(np.mean(fitnesses))
    varT = float(np.var(fitnesses))
    rangT = maxT - minT
    corrOuts = corrFitness(fitnesses, latent_dists, latent_dists_encoder) if latents is not None else {}
    if addFitnesses:
        corrOuts = {**corrOuts, 'latent_matrices' : latents_to_pick, 'latent_matrices_encoder' : latents_to_pick_encoder,
                    'mutants_decoder': mutants_decoder_to_pick,
                    'fitnesses_decoder': fitness2_decoder}
    return {**corrOuts, **{'avg_f': avg, 'var_f': var, 'max_f': max, 'min_f': min, 'rang_f': rang, 'bad_mutants_f': bad_mutants,
            'min_raw': minT, 'max_raw':maxT, 'avg_raw':avgT, 'var_raw':varT, 'rang_raw':rangT, 'centroid_fitness': centroid_fitness,
                           'fitnesses':fitnesses, 'mutants':mutants_to_pick
            }}



def mutate_latent(centroid : np.ndarray, times, power, evolution_model : EvolutionModel, framsCLI : FramsticksCLI, original_centroid) -> Tuple[List[str], int, np.ndarray]:
    max_times = 500
    latent_mutants = evolution_model.mutate(np.repeat(centroid[np.newaxis, :], max_times, axis=0), 1, power)

    frams = evolution_model.decode_latents(latent_mutants)
    # print(frams)
    validity = checkValid(evolution_model, frams, valid_key='mutate')
    frams = [(fram, ind) for fram, valid, ind in zip(frams, validity, range(len(frams))) if valid]
    validLen = len(frams)
    frams_inds = [(fram,ind) for fram,ind in frams if fram != original_centroid]
    frams = [fram for fram,ind in frams_inds]
    inds = [ind for fram,ind in frams_inds]
    latent_mutants = latent_mutants[inds]
    acceptLen = len(frams)
    same_as_centroid = validLen-acceptLen
    print("mutants: valid/different than centroid: " + str(acceptLen) + ", need: " + str(times))
    if acceptLen == 0:
        raise BadCentroidException("Bad centroid")
    if acceptLen>= times:
        return frams[:times], same_as_centroid, latent_mutants[:times]
    else:
        frams2, same_as_centroid2, latent_mutants2 = mutate_latent(centroid, times - acceptLen, power, evolution_model, framsCLI, original_centroid)
        return frams + frams2, same_as_centroid+same_as_centroid2, np.concatenate([latent_mutants, latent_mutants2])

def calculateMutateDistance(centroid : str, times, power, distance, evolutionModel : EvolutionModel, centroid_latent, representation='latent', framsCLI=None) -> Dict:

    if representation == 'latent':
        print(centroid)
        mutants, same_as_centroid, latent_mutants = mutate_latent(centroid_latent, times, power, evolutionModel, framsCLI, centroid)
        # same_as_centroid = len([mut for mut in mutants if mut == centroid])
        duplicates = len(mutants) - len(set(mutants))
        latent_mutants_encoder = evolutionModel.predictEncoderLatent(
            prepareGenos(['S' + mutant + 'T' for mutant in mutants], evolutionModel.config)[0])
        mutants_decoder = evolutionModel.decode_latents(latent_mutants_encoder)


    else:
        raise NotImplementedError()

    outputs = {'same_as_centroid':same_as_centroid, 'duplicates':duplicates}

    if 'l' in distance:
        outputs.update(distanceLevenhstein(centroid, mutants, latent_mutants, latent_mutants_encoder))
    if 'd' in distance:
        outputs.update(distanceDissim(centroid, mutants, framsCLI, evolutionModel, latent_mutants, latent_mutants_encoder))
    if 'f' in distance:
        try:
            outputs.update(distanceFitness(centroid, mutants, framsCLI, evolutionModel, latent_mutants, latent_mutants_encoder, addFitnesses=True if 'a' in distance else False, mutants_decoder=mutants_decoder if 'z' in distance else None))
        except BadCentroidException as e:
            outputs['bad_centroid_f'] = 1


    return outputs

def getCentroids_latent(num_centroids, evolution_model : EvolutionModel, framsCLI : FramsticksCLI, not_accept_centroids:List[str] = None) -> Tuple[List[str], int, np.ndarray]:
    if not_accept_centroids is None:
        not_accept_centroids = []

    max_times = 500
    samples = evolution_model.sampleMultivariateGaussianLatent(max_times, power=evolution_model.config['initial_sample_power'])

    frams = evolution_model.decode_latents(samples)
    # print(frams)
    # validity = framsCLI.isValid([evolution_model.config['prefix'] + fram for fram in frams], 'centroid')
    validity = checkValid(evolution_model, frams, valid_key='centroid')
    frams = [(fram, ind) for fram, valid, ind in zip(frams, validity, range(len(frams))) if valid]
    validLen = len(frams)
    current_frams = [fram for fram,ind in frams]
    frams_inds = [(fram, ind) for index, (fram, ind) in enumerate(frams) if fram not in current_frams[index+1:] and fram not in not_accept_centroids]
    acceptedLen = len(frams)
    duplicateCentroids = validLen-acceptedLen

    frams = [fram for fram, ind in frams_inds]
    inds = [ind for fram, ind in frams_inds]
    samples = samples[inds]


    print("centroids: valid and not duplicated: " + str(acceptedLen) + ", need: " + str(num_centroids))
    if validLen >= num_centroids:
        return frams[:num_centroids], duplicateCentroids, samples[:num_centroids]
    else:
        frams2, duplicateCentroids2, samples2 = getCentroids_latent(num_centroids-acceptedLen, evolution_model, framsCLI, not_accept_centroids+frams)
        return frams + frams2, duplicateCentroids + duplicateCentroids2, np.concatenate([samples, samples2])
import re
def prep_genos(genos, repr='f1'):
    prefix=''
    if repr == 'f9':
        prefix = '/*9*/'
    if repr == 'f4':
        prefix = '/*4*/'

    return [prefix + re.sub('[ST]', '', gen) if not gen.startswith(prefix) else re.sub('[ST]', '', gen) for gen in genos]
    # return [prefix+gen[1:-1] for gen in genos]
def mutateF(framsCLI : FramsticksCLI, evolutionModel, centroid, times, diversity, level=0):
    try:
        return framsCLI.ask_for_genos(prep_genos([centroid], repr=evolutionModel.config['representation']), times, diversity, 30)
    except:
        print('mutateF error level: ', level)
        return mutateF(framsCLI, evolutionModel, centroid, times, diversity, level+1)

def get_mutants_f9(framsCLI : FramsticksCLI, evolutionModel, centroid, times, diversity, level=0):
    mutants = []
    while len(mutants) < times:
        mutant = prep_genos([centroid], repr=evolutionModel.config['representation'])[0]
        for i in range(diversity):
            mutant = framsCLI.mutate(mutant)
        mutants.append(mutant)
    return mutants




def calculateMutDistF(centroid : str, times, distance, power, evolutionModel : EvolutionModel, framsCLI : FramsticksCLI=None):
    # mutants = []
    # power = int(power*40)
    if evolutionModel.config['representation'] == 'f1':
        mutants = mutateF(framsCLI, evolutionModel, centroid, times, power)
    elif evolutionModel.config['representation'] == 'f9':
        mutants = get_mutants_f9(framsCLI, evolutionModel, centroid, times, power)
    else:
        raise NotImplementedError()
    mutants = [mutant.replace(evolutionModel.config['prefix'], '') for mutant in mutants]
    duplicates = len(mutants) - len(set(mutants))
    same_as_centroid = len([mut for mut in mutants if mut == centroid])
    outputs = {'same_as_centroid': same_as_centroid, 'duplicates': duplicates}

    if 'l' in distance:
        outputs.update(distanceLevenhstein(centroid, mutants))
    if 'd' in distance:
        outputs.update(
            distanceDissim(centroid, mutants, framsCLI, evolutionModel))
    if 'f' in distance:
        try:
            outputs.update(
                distanceFitness(centroid, mutants, framsCLI, evolutionModel))
        except BadCentroidException as e:
            outputs['bad_centroid_f'] = 1

    return outputs

def calculateMutateDistancesF(num_centroids, times, distance, power, evolutionModel:EvolutionModel, framsCLI=None):
    centroids = np.random.choice(evolutionModel.genos, num_centroids)
    outputses = []
    fitnesses = []
    mutants = []
    stats = defaultdict(list)
    for centr_ind, centroid in enumerate(centroids):
        outputs = calculateMutDistF(centroid[1:-1], times, distance, power, evolutionModel, framsCLI)
        print("Centroid outputs: ", outputs, flush=True)

        fitnesses.extend(outputs['fitnesses'])
        mutants.extend(outputs['mutants'])
        del outputs['fitnesses']
        del outputs['mutants']
        outputses.append(outputs)
        for k, v in outputs.items():
            stats[k].append(v)

    d = {'power':power, 'fitnesses':fitnesses, 'mutants':mutants}
    for k,v in stats.items():
        d[k+'_mean'] = np.nanmean(v)
        d[k+'_median'] = np.nanmedian(v)
        d[k + '_max'] = np.nanmax(v)
        d[k + '_min'] = np.nanmin(v)
    return d

def dissimCentroids(evolutionModel, centroids, level=0):
    try:
        return evolutionModel.framsCLI.dissimilarity([evolutionModel.config['prefix']+ centroid for centroid in centroids])
    except:
        print("error, disssimCentroids, level ", level)
        return dissimCentroids(evolutionModel, centroids, level+1)

def calculateMutateDistances(num_centroids, times, power, distance, evolutionModel : EvolutionModel, representation='latent', framsCLI = None):
    centroids, duplicates_centroids, latent_centroids = getCentroids_latent(num_centroids, evolutionModel, framsCLI)

    latent_centroids_encoder = evolutionModel.predictEncoderLatent(prepareGenos(['S'+centroid+'T' for centroid in centroids], evolutionModel.config)[0])


    latent_centroids_dists = pdist(latent_centroids, 'euclidean')
    latent_centroids_dists_encoder = pdist(latent_centroids_encoder, 'euclidean')
    corrLevOutputs = corrLevenshtein(centroids, latent_centroids_dists, latent_centroids_dists_encoder, suffix='_centroids')
    dissim = dissimCentroids(evolutionModel, centroids)
    corrDissimOutputs = corrDissim([dissim], [latent_centroids_dists], [latent_centroids_dists_encoder], suffix='_centroids')


    fitness, bad_mutants_temp, good_inds = evolutionModel.fitness_check(centroids, True)
    fitness = list(np.array(fitness)[good_inds])
    latent_centroids_fitness = latent_centroids[good_inds]
    latent_centroids_fitness_encoder = latent_centroids_encoder[good_inds]

    latent_centroids_dists_fitness = pdist(latent_centroids_fitness, 'euclidean')
    latent_centroids_dists_fitness_encoder = pdist(latent_centroids_fitness_encoder, 'euclidean')
    corrFitnessOutputs= corrFitness(fitness, latent_centroids_dists_fitness, latent_centroids_dists_fitness_encoder, suffix='_centroids')

    # print(centroids)
    stats = defaultdict(list)
    same_as_centroids = []
    duplicatess = []
    bad_mutants_d = []
    bad_mutants_f = []
    inv_centroids = evolutionModel.invalidCnt['centroid']
    valid_centroids = evolutionModel.validCnt['centroid']
    sum_centroids = inv_centroids+valid_centroids

    bad_centroids = 0
    bad_centroids_f = 0

    inv_mutants = []
    valid_mutants = []
    outputses = []

    latents = None
    latents_encoder = None
    mutants = []
    mutants_decoder = []
    fitnesses_decoder = []
    fitnesses=[]
    for centr_ind, centroid in enumerate(centroids):
        framsCLI.invalidCnt = Counter()
        framsCLI.validCnt = Counter()
        try:
            outputs = calculateMutateDistance(centroid, times, power, distance, evolutionModel, latent_centroids_encoder[centr_ind], representation, framsCLI)


            fitnesses.extend(outputs['fitnesses'])
            mutants.extend(outputs['mutants'])
            del outputs['fitnesses']
            del outputs['mutants']
            # print("Centroid outputs: ", outputs, flush=True)
            if 'd' in distance:
                bad_mutants_d.append(outputs['bad_mutants_d'])
                del outputs['bad_mutants_d']
            if 'f' in distance:
                if 'bad_centroid_f' in outputs:
                    bad_centroids_f +=1
                else:
                    bad_mutants_f.append(outputs['bad_mutants_f'])
                    del outputs['bad_mutants_f']

            if 'a' in distance:
                if 'latent_matrices' in outputs:
                    if latents is None:
                        latents = outputs['latent_matrices']
                        latents_encoder = outputs['latent_matrices_encoder']
                    else:
                        latents = np.concatenate([latents, outputs['latent_matrices']], axis=0)
                        latents_encoder = np.concatenate([latents_encoder, outputs['latent_matrices_encoder']], axis=0)


                del outputs['latent_matrices']
                del outputs['latent_matrices_encoder']

                if 'mutants_decoder' in outputs:
                    mutants_decoder.extend(outputs['mutants_decoder'])
                    del outputs['mutants_decoder']
                    fitnesses_decoder.extend(outputs['fitnesses_decoder'])
                    del outputs['fitnesses_decoder']
            same_as_centroids.append(outputs['same_as_centroid'])
            del outputs['same_as_centroid']
            duplicatess.append(outputs['duplicates'])
            del outputs['duplicates']

            inv_mutants.append(evolutionModel.invalidCnt['mutate'])
            valid_mutants.append(evolutionModel.validCnt['mutate'])
            outputses.append(outputs)
            for k,v in outputs.items():
                stats[k].append(v)



        except BadCentroidException as e:
            bad_centroids += 1

    total_inv_mut = np.sum(inv_mutants)
    total_valid_mut = np.sum(valid_mutants)
    total_same_as_centr = np.sum(same_as_centroids)
    total_duplicates = np.sum(duplicatess)
    total_bad_mutants_d = np.sum(bad_mutants_d)
    total_bad_mutants_f = np.sum(bad_mutants_f)
    for i in range(len(inv_mutants)):
        sum_mut = inv_mutants[i] + valid_mutants[i]
        inv_mutants[i] /= sum_mut
        valid_mutants[i] /= sum_mut

    d = {**{'epochs_train_total': evolutionModel.eps_tot[-1],
         'accuracy': evolutionModel.accs[-1],
        'num_centroids': num_centroids,
         'num_mutants':times,
         'power':power,
         'distance_type':distance,
         'representation_type':representation,
         'total_invalid_mutants':total_inv_mut,
         'total_valid_mutants': total_valid_mut,
         'total_same_as_centroids':total_same_as_centr,
         'total_duplicates': total_duplicates,
         'same_as_centroids_rate': np.mean(same_as_centroids)/times,
         'duplicate_rate': np.mean(duplicatess)/times,
         'bad_centroids': bad_centroids,
         'bad_centroids_rate': bad_centroids/num_centroids,
         'invalid_centroids':inv_centroids/sum_centroids,
         'valid_centroids':valid_centroids/sum_centroids,
         'invalid_mutants': np.mean(inv_mutants),
         'valid_mutants': np.mean(valid_mutants),
         'total_bad_mutants_d': total_bad_mutants_d,
         'total_bad_mutants_f': total_bad_mutants_f,
         'bad_mutants_d_rate': np.mean(bad_mutants_d)/times,
         'bad_mutants_f_rate': np.mean(bad_mutants_f) / times,
         'bad_centroids_f': bad_centroids_f,
         'duplicate_centroids': duplicates_centroids,
         'outputses': outputses,
         'latents': latents,
         'latents_encoder':latents_encoder,
         'fitnesses':fitnesses,
         'mutants' : mutants,
         'mutants_decoder_' : mutants_decoder,
         'fitnesses_decoder' : fitnesses_decoder,
        'test':evolutionModel.config['test'],
            'config':evolutionModel.config,
            'fake_fitness': fake_fitness[0]
         }, **corrDissimOutputs, **corrLevOutputs, **corrFitnessOutputs}
    for k,v in stats.items():
        d[k+'_mean'] = np.nanmean(v)
        d[k+'_median'] = np.nanmedian(v)
        d[k + '_max'] = np.nanmax(v)
        d[k + '_min'] = np.nanmin(v)
    return d

def save_to_shared_directory(config, result, name):
    print("Saving")
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

    framsCLI=createCLI(config)
    # framsCLI = FramsticksCLI(config['frams_path'], None, pid=config['model_name'])#, importSim=importSim)
    results = []
    for powerr in config['calculateMutDistPowers']:
        if 'calculateOriginalRepresentation' not in config or not config['calculateOriginalRepresentation']:

            res = calculateMutateDistances(config['calculateNumCentroids'], config['calculateNumMutants'], powerr, 'ldfaz', EM, 'latent', framsCLI)
        else:
            res = calculateMutateDistancesF(config['calculateNumCentroids'], config['calculateNumMutants'], 'ldfaz', powerr, EM, framsCLI)

        print(res)
        results.append(res)
        save_to_shared_directory(config, results, 'mutDist' if 'mutDistName' not in config else config['mutDistName'])



