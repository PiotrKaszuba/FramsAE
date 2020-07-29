import os

import numpy as np
from deap import creator, base, tools, algorithms

from Code.FramsticksCli import FramsticksCLI
import glob
import pickle
from Code.Preparation.Run_preparation_slurm import EvolutionModel, prepareGenos, extractIthGenotypeFromEncoderPred, splitLatentToHiddenAndCell
from Code.Preparation.configuration import get_config
from Code.Preparation.Utils import extract_fram
from Code.Preparation.Run_preparation_slurm import prepareData
# Note: this is much less efficient than running the evolution directly in Framsticks, so use only when required or when poor performance is acceptable!


# The list of criteria includes 'vertpos', 'velocity', 'distance', 'vertvel', 'lifespan', 'numjoints', 'numparts', 'numneurons', 'numconnections'.
OPTIMIZATION_CRITERIA = [
    'vertpos']  # Single or multiple criteria. Names from the standard-eval.expdef dictionary, e.g. ['vertpos', 'velocity'].


def frams_evaluate_lat(frams_cli, evolution_model : EvolutionModel, individual):

    hidden, cell = splitLatentToHiddenAndCell(individual)
    pred = evolution_model.decoder.predict([hidden, cell, evolution_model.getOnesMaskForFullPrediction(1)])
    genos = evolution_model.get_genos(pred)
    fram = extract_fram(genos[0])




    data = frams_cli.evaluate(fram)
    # print("Evaluated '%s'" % genotype, 'evaluation is:', data)
    try:
        first_genotype_data = data[0]
        evaluation_data = first_genotype_data["evaluations"]
        default_evaluation_data = evaluation_data[""]
        fitness = [default_evaluation_data[crit] for crit in OPTIMIZATION_CRITERIA]
    except (KeyError, TypeError) as e:  # the evaluation may have failed for invalid genotypes (or some other reason)
        fitness = [-1] * len(OPTIMIZATION_CRITERIA)
        print("Error '%s': could not evaluate genotype '%s', returning fitness %s" % (str(e), fram, fitness))
    return fitness


def frams_crossover_lat(frams_cli, evolution_model, individual1, individual2):
    inner_ind1 = np.copy(individual1[0])
    inner_ind2 = np.copy(individual2[0])
    inner_ind1_expand = np.expand_dims(inner_ind1, axis=0)
    inner_ind2_expand = np.expand_dims(inner_ind2, axis=0)

    conc = np.concatenate([inner_ind1_expand, inner_ind2_expand], axis=0)




    individual_temp = np.mean(conc, axis=0)

    hidden, cell = splitLatentToHiddenAndCell(np.expand_dims(individual_temp, axis=0))
    pred = evolution_model.decoder.predict([hidden, cell, evolution_model.getOnesMaskForFullPrediction(1)])
    genos = evolution_model.get_genos(pred)
    fram = extract_fram(genos[0])
    valid = frams_cli.isValid(fram)
    if valid:
        print("Good cross: " + str(fram))
        individual1[0] = individual_temp
        # individual2[0] = individual_temp
    else:
        print("Invalid cross: "+ str(fram))


    return individual1, individual2

def get_fram(individual_temp, evolution_model):
    hidden, cell = splitLatentToHiddenAndCell(individual_temp)
    pred = evolution_model.decoder.predict([hidden, cell, evolution_model.getOnesMaskForFullPrediction(1)])
    genos = evolution_model.get_genos(pred)
    fram = extract_fram(genos[0])
    return fram
def frams_mutate_lat(frams_cli, evolution_model : EvolutionModel, individual):
    inner_ind = np.copy(individual[0])
    for i in range(5):
        magnitude=0.005*np.random.randint(1,6)
        # if i > 0:
        #     magnitude *= 2*i
        individual_temp = evolution_model.mutate(np.array([inner_ind]), times=1, magnitude=magnitude)

        fram = get_fram(individual_temp, evolution_model)
        valid = frams_cli.isValid(fram)
        if valid:
            print("Good mutation: " + str(fram))
            individual[0] = individual_temp
            break
        else:
            print("Invalid mutation " + str(i) +", fram: " + str(fram))

    return individual,


def frams_getsimplest_lat(frams_cli, genetic_format, evolution_model):

    # if evolution_model.simplest is None:
    #
    #     # fram =  frams_cli.getSimplest(genetic_format)
    #     # genos = [fram]
    #     # X, Y, genosCheck = prepareGenos(genos, evolution_model.config)
    #     # latent = evolution_model.encoder.predict(X)
    #     # simplest = extractIthGenotypeFromEncoderPred(latent, 0)
    #     evolution_model


    max = np.shape(evolution_model.latent)[0]
    return evolution_model.latent[np.random.randint(max)]


def frams_evaluate(frams_cli, individual):
    genotype = individual[
        0]  # [0] because we can't (?) have a simple str as a deap genotype/individual, only list of str.
    data = frams_cli.evaluate(genotype)
    # print("Evaluated '%s'" % genotype, 'evaluation is:', data)
    try:
        first_genotype_data = data[0]
        evaluation_data = first_genotype_data["evaluations"]
        default_evaluation_data = evaluation_data[""]
        fitness = [default_evaluation_data[crit] for crit in OPTIMIZATION_CRITERIA]
    except (KeyError, TypeError) as e:  # the evaluation may have failed for invalid genotypes (or some other reason)
        fitness = [-1] * len(OPTIMIZATION_CRITERIA)
        print("Error '%s': could not evaluate genotype '%s', returning fitness %s" % (str(e), genotype, fitness))
    return fitness


def frams_crossover(frams_cli, individual1, individual2):
    geno1 = individual1[
        0]  # [0] because we can't (?) have a simple str as a deap genotype/individual, only list of str.
    geno2 = individual2[
        0]  # [0] because we can't (?) have a simple str as a deap genotype/individual, only list of str.
    individual1[0] = frams_cli.crossOver(geno1, geno2)
    individual2[0] = frams_cli.crossOver(geno1, geno2)
    return individual1, individual2


def frams_mutate(frams_cli, individual):
    individual[0] = frams_cli.mutate(individual[0])
    return individual,


def frams_getsimplest(frams_cli, genetic_format, genos):
    max = len(genos)
    return extract_fram(genos[np.random.randint(max)])


def prepareToolbox(frams_cli, latent, genetic_format, evolution_model, genos, rand):
    creator.create("FitnessMax", base.Fitness, weights=[1.0] * len(OPTIMIZATION_CRITERIA))
    creator.create("Individual", np.ndarray,
                   fitness=creator.FitnessMax)  # would be nice to have "str" instead of unnecessary "list of str"
    toolbox = base.Toolbox()
    if latent == 'nolatent':
        simplest = frams_getsimplest
        eval = frams_evaluate
        cross = frams_crossover
        mut = frams_mutate
        additional_arg = genos
    else:
        simplest = frams_getsimplest_lat
        eval = frams_evaluate_lat
        cross = frams_crossover_lat
        mut = frams_mutate_lat
        additional_arg = evolution_model

    toolbox.register("attr_simplest_genotype", simplest, frams_cli, genetic_format,
                     additional_arg)  # "Attribute generator"
    # (failed) struggle to have an individual which is a simple str, not a list of str
    # toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_frams)
    # https://stackoverflow.com/questions/51451815/python-deap-library-using-random-words-as-individuals
    # https://github.com/DEAP/deap/issues/339
    # https://gitlab.com/santiagoandre/deap-customize-population-example/-/blob/master/AGbasic.py
    # https://groups.google.com/forum/#!topic/deap-users/22g1kyrpKy8
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_simplest_genotype, 1)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", eval, frams_cli, additional_arg)
    toolbox.register("mate", cross, frams_cli, additional_arg)
    toolbox.register("mutate", mut, frams_cli, additional_arg)
    if len(OPTIMIZATION_CRITERIA) == 1:
        toolbox.register("select", tools.selTournament, tournsize=3 if not rand else 1)
    else:
        toolbox.register("select", tools.selNSGA2)
    return toolbox




def create_config(data_path, load_dir):
    # representation = 'f1'
    # cells = 64
    # long_genos = None
    # twoLayer = 'oneLayer'
    # bidir = 'Bidir'
    #
    #


    data_path = list(os.path.split(data_path))
    params = data_path[1].split('_')
    model_name = '_'.join(params[1:-4])
    representation = params[1]
    cells =int(params[3])
    long_genos = None
    twoLayer = params[4]
    bidir = params[5]
    loc = params[6]
    tes = params[7]
    model_name = 'model_' + representation + '_' + str(long_genos) + '_' + str(cells) + '_' + twoLayer + '_' + bidir + '_' + loc + '_' + tes
    print(model_name)
    data_path = os.path.join(data_path[0], model_name)
    max_len = 100
    config = get_config(model_name, representation, long_genos, cells, twoLayer, bidir, data_path, load_dir,
                        max_len=max_len, locality=loc, test=tes)
    # config['features'] = 21

    return config


def runEvolLatent(evol_name, representation, data_path, load_dir, frams_path, rand : bool, gene, pop_s, latent):
    # A demo run: optimize OPTIMIZATION_CRITERIA
    # find max logbook
    evol_path_name = os.path.join(data_path, evol_name)
    iteration = -1
    files = []
    for filename in glob.glob(os.path.join(data_path, "logbook_*")):
        try:
            files.append((filename, int(filename.split('_')[-1])))
        except:
            pass

    files = list(set(files))

    if len(files) > 0:
        files = sorted(files, key=lambda x: x[1], reverse=True)
        iteration = int(files[0][1])
    # random.seed(123)  # see FramsticksCLI.DETERMINISTIC below, set to True if you want full determinism

    config = create_config(data_path, load_dir)
    EM = EvolutionModel(config)
    genos = prepareData(config)[2]
    FramsticksCLI.DETERMINISTIC = False  # must be set before FramsticksCLI() constructor call
    representation = representation[1:]

    importSim = 'Simulator.import(\"generation_params.sim\");' if latent == 'nolatent' else None
    framsCLI = FramsticksCLI(frams_path, None, pid=evol_name, importSim=importSim)

    framsCLI.PRINT_FRAMSTICKS_OUTPUT = True

    framsCLI.PRINT_FRAMSTICKS_OUTPUT = False
    toolbox = prepareToolbox(framsCLI, latent, '1' if representation is None else representation,
                             evolution_model=EM, genos=genos, rand=rand)

    POPSIZE = pop_s
    GENERATIONS = gene
    iterations = 10
    import sys



    while iteration < iterations-1:
        iteration += 1
        with open(os.path.join(data_path,'experiment_outFile_%s' % str(iteration)), 'w') as outF:
            sys.stdout = outF


            print("ITERATION: " + str(iteration))

            pop = toolbox.population(n=POPSIZE)
            if latent == 'nolatent':
                hof = tools.HallOfFame(10)
            else:
                hof = tools.HallOfFame(10, similar=np.allclose)
            stats = tools.Statistics(lambda ind: ind.fitness.values)
            stats.register("avg", np.mean)
            stats.register("stddev", np.std)
            stats.register("min", np.min)
            stats.register("max", np.max)

            print('Evolution with population size %d for %d generations, optimization criteria: %s' % (
            POPSIZE, GENERATIONS, OPTIMIZATION_CRITERIA))

            logbook = tools.Logbook()

            def runAlg(pop, log):
                try:
                    pop, log = algorithms.eaSimple(pop, toolbox, cxpb=0.15, mutpb=0.85, ngen=GENERATIONS, stats=stats,
                                                   halloffame=hof,
                                                   verbose=True)
                except Exception as e:
                    print(e)
                    print("Error! Stopping!")
                return pop, log


            pop, logbook = runAlg(pop, logbook)

            with open(os.path.join(data_path, 'logbook_%s' % str(iteration)), 'wb') as f:
                pickle.dump(logbook, f)

            print('Best individuals:')
            for best in hof:
                if latent == 'nolatent':
                    print(best.fitness, '\t-->\t', best[0])
                else:
                    fram = get_fram(best, EM)
                    print(best.fitness, '\t-->\t', fram)
                    print(best[0])
