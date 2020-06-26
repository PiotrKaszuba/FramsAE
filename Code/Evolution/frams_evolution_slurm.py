import argparse
import glob
import os
import pickle
import sys

import numpy as np
from FramsticksCLI_importSim import FramsticksCLI
from configuration import get_config
from Run_preparation_slurm import prepareData
from Utils import extract_fram
from deap import creator, base, tools, algorithms

# Note: this is much less efficient than running the evolution directly in Framsticks, so use only when required or when poor performance is acceptable!


# The list of criteria includes 'vertpos', 'velocity', 'distance', 'vertvel', 'lifespan', 'numjoints', 'numparts', 'numneurons', 'numconnections'.
OPTIMIZATION_CRITERIA = [
    'vertpos']  # Single or multiple criteria. Names from the standard-eval.expdef dictionary, e.g. ['vertpos', 'velocity'].


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


def prepareToolbox(frams_cli, genetic_format, genos):
    creator.create("FitnessMax", base.Fitness, weights=[1.0] * len(OPTIMIZATION_CRITERIA))
    creator.create("Individual", list,
                   fitness=creator.FitnessMax)  # would be nice to have "str" instead of unnecessary "list of str"

    toolbox = base.Toolbox()
    toolbox.register("attr_simplest_genotype", frams_getsimplest, frams_cli, genetic_format, genos)  # "Attribute generator"
    # (failed) struggle to have an individual which is a simple str, not a list of str
    # toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_frams)
    # https://stackoverflow.com/questions/51451815/python-deap-library-using-random-words-as-individuals
    # https://github.com/DEAP/deap/issues/339
    # https://gitlab.com/santiagoandre/deap-customize-population-example/-/blob/master/AGbasic.py
    # https://groups.google.com/forum/#!topic/deap-users/22g1kyrpKy8
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_simplest_genotype, 1)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", frams_evaluate, frams_cli)
    toolbox.register("mate", frams_crossover, frams_cli)
    toolbox.register("mutate", frams_mutate, frams_cli)
    if len(OPTIMIZATION_CRITERIA) == 1:
        toolbox.register("select", tools.selTournament, tournsize=3)
    else:
        toolbox.register("select", tools.selNSGA2)
    return toolbox


def parseArguments():
    parser = argparse.ArgumentParser(
        description='Train this program with "python -u %s" if you want to disable buffering of its output.' % sys.argv[
            0])
    parser.add_argument('-path', type=ensureDir, required=True, help='Path to Framsticks CLI without trailing slash.')
    parser.add_argument('-exe', required=False,
                        help='Executable name. If not given, "frams.exe" or "frams.linux" is assumed.')
    parser.add_argument('-genformat', required=False,
                        help='Genetic format for the demo run, for example 4, 9, or B. If not given, f1 is assumed.')
    return parser.parse_args()


def ensureDir(string):
    if os.path.isdir(string):
        return string
    else:
        raise NotADirectoryError(string)


def create_config(data_path, load_dir):
    representation = 'f1'
    cells = 64
    long_genos = None
    twoLayer = 'oneLayer'
    bidir = 'Bidir'

    model_name = 'model_' + representation + '_' + str(long_genos) + '_' + str(cells) + '_' + twoLayer + '_' + bidir

    data_path = list(os.path.split(data_path))

    data_path = os.path.join(data_path[0], model_name)
    max_len = 100
    config = get_config(model_name, representation, long_genos, cells, twoLayer, bidir, data_path, load_dir,
                        max_len=max_len)
    # config['features'] = 21

    return config


def runEvol(evol_name, representation, data_path, load_dir, frams_path):
    config = create_config(data_path, load_dir)
    genos = prepareData(config)[2]
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
    FramsticksCLI.DETERMINISTIC = False  # must be set before FramsticksCLI() constructor call
    representation = representation[1:]
    framsCLI = FramsticksCLI(frams_path, None, outer_writing_path=None)
    framsCLI.PRINT_FRAMSTICKS_OUTPUT = True
    framsCLI.rawCommand('Simulator.import(\"generation_params.sim\");')
    framsCLI.PRINT_FRAMSTICKS_OUTPUT = False
    toolbox = prepareToolbox(framsCLI, '1' if representation is None else representation, genos=genos)

    POPSIZE = 50
    GENERATIONS = 1000

    iterations = 10
    import sys

    while iteration < iterations - 1:
        iteration += 1
        with open(os.path.join(data_path, 'experiment_outFile_%s' % str(iteration)), 'w') as outF:
            sys.stdout = outF

            print("ITERATION: " + str(iteration))

            pop = toolbox.population(n=POPSIZE)
            hof = tools.HallOfFame(10)
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
                print(best.fitness, '\t-->\t', best[0])
