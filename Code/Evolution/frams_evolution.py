import argparse
import os
import sys
import numpy as np
from deap import creator, base, tools, algorithms
from Code.FramsticksCLI import FramsticksCLI

# Note: this is much less efficient than running the evolution directly in Framsticks, so use only when required or when poor performance is acceptable!


# The list of criteria includes 'vertpos', 'velocity', 'distance', 'vertvel', 'lifespan', 'numjoints', 'numparts', 'numneurons', 'numconnections'.
OPTIMIZATION_CRITERIA = ['vertpos']  # Single or multiple criteria. Names from the standard-eval.expdef dictionary, e.g. ['vertpos', 'velocity'].


def frams_evaluate(frams_cli, individual):
	genotype = individual[0]  # [0] because we can't (?) have a simple str as a deap genotype/individual, only list of str.
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
	geno1 = individual1[0]  # [0] because we can't (?) have a simple str as a deap genotype/individual, only list of str.
	geno2 = individual2[0]  # [0] because we can't (?) have a simple str as a deap genotype/individual, only list of str.
	individual1[0] = frams_cli.crossOver(geno1, geno2)
	individual2[0] = frams_cli.crossOver(geno1, geno2)
	return individual1, individual2


def frams_mutate(frams_cli, individual):
	individual[0] = frams_cli.mutate(individual[0])
	return individual,


def frams_getsimplest(frams_cli, genetic_format):
	return frams_cli.getSimplest(genetic_format)


def prepareToolbox(frams_cli, genetic_format):
	creator.create("FitnessMax", base.Fitness, weights=[1.0] * len(OPTIMIZATION_CRITERIA))
	creator.create("Individual", list, fitness=creator.FitnessMax)  # would be nice to have "str" instead of unnecessary "list of str"

	toolbox = base.Toolbox()
	toolbox.register("attr_simplest_genotype", frams_getsimplest, frams_cli, genetic_format)  # "Attribute generator"
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
		toolbox.register("select", tools.selTournament, tournsize=5)
	else:
		toolbox.register("select", tools.selNSGA2)
	return toolbox


def parseArguments():
	parser = argparse.ArgumentParser(description='Train this program with "python -u %s" if you want to disable buffering of its output.' % sys.argv[0])
	parser.add_argument('-path', type=ensureDir, required=True, help='Path to Framsticks CLI without trailing slash.')
	parser.add_argument('-exe', required=False, help='Executable name. If not given, "frams.exe" or "frams.linux" is assumed.')
	parser.add_argument('-genformat', required=False, help='Genetic format for the demo run, for example 4, 9, or B. If not given, f1 is assumed.')
	return parser.parse_args()


def ensureDir(string):
	if os.path.isdir(string):
		return string
	else:
		raise NotADirectoryError(string)


if __name__ == "__main__":
	# A demo run: optimize OPTIMIZATION_CRITERIA

	# random.seed(123)  # see FramsticksCLI.DETERMINISTIC below, set to True if you want full determinism
	FramsticksCLI.DETERMINISTIC = False  # must be set before FramsticksCLI() constructor call
	representation = '1'
	framsCLI = FramsticksCLI('C:/Users/Piotr/Desktop/Framsticks50rc14', None)

	toolbox = prepareToolbox(framsCLI, '1' if representation is None else representation)

	POPSIZE = 30
	GENERATIONS = 100

	pop = toolbox.population(n=POPSIZE)
	hof = tools.HallOfFame(5)
	stats = tools.Statistics(lambda ind: ind.fitness.values)
	stats.register("avg", np.mean)
	stats.register("stddev", np.std)
	stats.register("min", np.min)
	stats.register("max", np.max)

	print('Evolution with population size %d for %d generations, optimization criteria: %s' % (POPSIZE, GENERATIONS, OPTIMIZATION_CRITERIA))
	pop, log = algorithms.eaSimple(pop, toolbox, cxpb=0.2, mutpb=0.8, ngen=GENERATIONS, stats=stats, halloffame=hof, verbose=True)
	print('Best individuals:')
	for best in hof:
		print(best.fitness, '\t-->\t', best[0])