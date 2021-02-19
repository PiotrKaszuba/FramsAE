import os
import sys
import numpy as np
from deap import creator, base, tools, algorithms
from deap.algorithms import varAnd
from Code.FramsticksCli import FramsticksCLI
import glob
import pickle
from Code.Preparation.Run_preparation_slurm import EvolutionModel, evaluate_with_batches
from Code.Preparation.Utils import extract_fram, prepareGenos
from Code.Preparation.locality_term_prep import createCLI
from Code.FramsticksCli import fake_fitness, fitness_min_value, fake_mutate, fitness_max_len, fitness_len_max_value, fitness_len_weight, fitness_len_chars_sub
from operator import attrgetter
import random
# Note: this is much less efficient than running the evolution directly in Framsticks, so use only when required or when poor performance is acceptable!


# The list of criteria includes 'vertpos', 'velocity', 'distance', 'vertvel', 'lifespan', 'numjoints', 'numparts', 'numneurons', 'numconnections'.
OPTIMIZATION_CRITERIA = [
    'vertpos']  # Single or multiple criteria. Names from the standard-eval.expdef dictionary, e.g. ['vertpos', 'velocity'].


def frams_evaluate_lat(evolution_model : EvolutionModel, individual):
    fram = evolution_model.decode_latents(np.expand_dims(individual[0], axis=0))[0]
    # print("---------------------")
    # print(fram)
    try:
        data = evolution_model.framsCLI.evaluate(evolution_model.config['prefix']+fram)
    except:
        evolution_model.createCLI()
    # print("Evaluated '%s'" % genotype, 'evaluation is:', data)
    try:
        first_genotype_data = data[0]
        evaluation_data = first_genotype_data["evaluations"]
        default_evaluation_data = evaluation_data[""]
        fitness = [default_evaluation_data[crit] for crit in OPTIMIZATION_CRITERIA]
        # print(fitness)
    except (KeyError, TypeError, NameError) as e:  # the evaluation may have failed for invalid genotypes (or some other reason)
        # print('err')
        fitness = [-1] * len(OPTIMIZATION_CRITERIA)
        print("Error '%s': could not evaluate genotype '%s', returning fitness %s" % (str(e), fram, fitness))
    # print("---------------")
    return fitness


def frams_crossover_lat(evolution_model : EvolutionModel, individual1, individual2):
    inner_ind1 = np.copy(individual1[0])
    inner_ind2 = np.copy(individual2[0])
    inner_ind1_expand = np.expand_dims(inner_ind1, axis=0)
    inner_ind2_expand = np.expand_dims(inner_ind2, axis=0)

    conc = np.concatenate([inner_ind1_expand, inner_ind2_expand], axis=0)

    max_tests = 200
    individuals_generated = []
    # differ1 = inner_ind1 - inner_ind2
    # differ2 = inner_ind2 - inner_ind1
    for i in range(max_tests):
        offspring_proxim1 = np.random.rand()
        individuals_generated.append(np.average(conc, axis=0, weights=[offspring_proxim1, 1 - offspring_proxim1]))

    frams = evolution_model.decode_latents(np.array(individuals_generated))

    already_seen = set()
    framszip = list(zip(frams, individuals_generated))
    framszip_set = [(fram,lat) for fram, lat in framszip if not (fram in already_seen or already_seen.add(fram))]
    frams, individuals_generated = list(zip(*framszip_set))

    # frams = list(set(frams))

    if evolution_model.config['evol_use_encoder']:
        genos_prep = evolution_model.get_input_from_genos(['S'+fram+'T' for fram in frams])

        individuals_generated = evolution_model.predictEncoderLatent(genos_prep)

        frams = evolution_model.decode_latents(individuals_generated)

        already_seen = set()
        framszip = list(zip(frams, individuals_generated))
        framszip_set = [(fram, lat) for fram, lat in framszip if not (fram in already_seen or already_seen.add(fram))]
        frams, individuals_generated = list(zip(*framszip_set))

    frams = [evolution_model.config['prefix'] + fram for fram in frams]
    try:
        valids = evolution_model.framsCLI.isValid(frams)

    except:
        evolution_model.createCLI()
        valids = []

    whereTrue = np.where(np.array(valids))[0]
    # print('whereTrue len: ', len(whereTrue))

    if len(whereTrue) > 0:

        indices = np.random.choice(whereTrue)
        whereTrue = np.delete(whereTrue, np.where(whereTrue == indices))
        individual1 = creator.Individual(( individuals_generated[indices],))
    if len(whereTrue) > 0:
        indices = np.random.choice(whereTrue)
        individual2=creator.Individual(( individuals_generated[indices],))

    # if valid:
    #     print("Good cross: " + str(fram))
    #     individual[0] = individual_temp
    # # individual2[0] = individual_temp
    # else:
    #     print("Invalid cross: "+ str(fram))
    # print(individual1, individual2)
    return individual1, individual2



def frams_crossover_lat2(evolution_model : EvolutionModel, individual1, individual2):
    inner_ind1 = np.copy(individual1[0])
    inner_ind2 = np.copy(individual2[0])
    # inner_ind1_expand = np.expand_dims(inner_ind1, axis=0)
    # inner_ind2_expand = np.expand_dims(inner_ind2, axis=0)
    #
    # conc = np.concatenate([inner_ind1_expand, inner_ind2_expand], axis=0)

    max_tests = 200
    individuals_generated = []
    differ1 = inner_ind1 - inner_ind2
    differ2 = inner_ind2 - inner_ind1
    for i in range(max_tests):
        offspring_proxim1 = np.random.rand()*1.66 - 0.66
        individuals_generated.append(inner_ind1+differ1*offspring_proxim1)
        individuals_generated.append(inner_ind2+differ2*offspring_proxim1)
        # individuals_generated.append(np.average(conc, axis=0, weights=[offspring_proxim1, 1 - offspring_proxim1]))

    frams = evolution_model.decode_latents(np.array(individuals_generated))

    already_seen = set()
    framszip = list(zip(frams, individuals_generated))
    framszip_set = [(fram,lat) for fram, lat in framszip if not (fram in already_seen or already_seen.add(fram))]
    frams, individuals_generated = list(zip(*framszip_set))

    # frams = list(set(frams))

    if evolution_model.config['evol_use_encoder']:
        genos_prep = evolution_model.get_input_from_genos(['S'+fram+'T' for fram in frams])

        individuals_generated = evolution_model.predictEncoderLatent(genos_prep)

        frams = evolution_model.decode_latents(individuals_generated)

        already_seen = set()
        framszip = list(zip(frams, individuals_generated))
        framszip_set = [(fram, lat) for fram, lat in framszip if not (fram in already_seen or already_seen.add(fram))]
        frams, individuals_generated = list(zip(*framszip_set))

    frams = [evolution_model.config['prefix'] + fram for fram in frams]
    try:
        valids = evolution_model.framsCLI.isValid(frams)

    except:
        evolution_model.createCLI()
        valids = []

    whereTrue = np.where(np.array(valids))[0]
    # print('whereTrue len: ', len(whereTrue))

    if len(whereTrue) > 0:

        indices = np.random.choice(whereTrue)
        whereTrue = np.delete(whereTrue, np.where(whereTrue == indices))
        individual1 = creator.Individual(( individuals_generated[indices],))
    if len(whereTrue) > 0:
        indices = np.random.choice(whereTrue)
        individual2 = creator.Individual(( individuals_generated[indices],))

    # if valid:
    #     print("Good cross: " + str(fram))
    #     individual[0] = individual_temp
    # # individual2[0] = individual_temp
    # else:
    #     print("Invalid cross: "+ str(fram))
    # print(individual1, individual2)
    return individual1, individual2








def frams_mutate_lat(evolution_model : EvolutionModel, individual):
    inner_ind = np.copy(individual[0])
    max_tests = 200
    # fram_org = evolution_model.decode_latents(np.expand_dims(inner_ind, axis=0))[0]
    magnitude = evolution_model.config['mut_magnitude'] #* np.random.randint(1, 6)
    individual_temps = evolution_model.mutate(np.array([inner_ind]*max_tests), times=1, magnitude=magnitude)
    frams = evolution_model.decode_latents(individual_temps)

    already_seen = set()
    framszip = list(zip(frams, individual_temps))
    framszip_set = [(fram, lat) for fram, lat in framszip if not (fram in already_seen or already_seen.add(fram))]
    frams, individuals_generated = list(zip(*framszip_set))

    if evolution_model.config['evol_use_encoder']:
        genos_prep = evolution_model.get_input_from_genos(['S'+fram+'T' for fram in frams])

        individuals_generated = evolution_model.predictEncoderLatent(genos_prep)

        frams = evolution_model.decode_latents(individuals_generated)
        already_seen = set()
        framszip = list(zip(frams, individual_temps))
        framszip_set = [(fram, lat) for fram, lat in framszip if not (fram in already_seen or already_seen.add(fram))]
        frams, individuals_generated = list(zip(*framszip_set))

    frams = [evolution_model.config['prefix']+fram for fram in frams]
    try:
        valids = evolution_model.framsCLI.isValid(frams)
        # valids = [valid for valid in valids]

    except:
        evolution_model.createCLI()
        valids = []
    # print('sum valids: ', sum(valids))
    if sum(valids) > 0:

        indices = np.random.choice(np.where(np.array(valids))[0])

        individual = creator.Individual((individuals_generated[indices],))

    # for i in range(5):
    #
    #     # if i > 0:
    #     #     magnitude *= 2*i
    #
    #
    #
    #     try:
    #         valid = frams_cli.isValid(fram)
    #     except:
    #         frams_cli[0] = createCLI(frams_cli[0].config, frams_cli[0])
    #         valid = False
    #     if valid:
    #         print("Good mutation: " + str(fram))
    #         individual[0] = individual_temp
    #         break
    #     else:
    #         print("Invalid mutation " + str(i) +", fram: " + str(fram))
    # print(individual)
    return individual,


def frams_getsimplest_lat(evolution_model, genetic_format, genos):

    # if evolution_model.simplest is None:
    #
    #     # fram =  frams_cli.getSimplest(genetic_format)
    #     # genos = [fram]
    #     # X, Y, genosCheck = prepareGenos(genos, evolution_model.config)
    #     # latent = evolution_model.encoder.predict(X)
    #     # simplest = extractIthGenotypeFromEncoderPred(latent, 0)
    #     evolution_model

    # return evolution_model.sampleMultivariateGaussianLatent(samples=1, power=5.0)[0]
    max = np.shape(evolution_model.latent)[0]
    return evolution_model.latent[np.random.randint(max)]


def frams_evaluate(evolution_model : EvolutionModel, individual):
    genotype = individual[
        0]  # [0] because we can't (?) have a simple str as a deap genotype/individual, only list of str.
    try:
        data = evolution_model.framsCLI.evaluate(genotype)
    except:
        evolution_model.createCLI()
    # print("Evaluated '%s'" % genotype, 'evaluation is:', data)
    try:
        first_genotype_data = data[0]
        evaluation_data = first_genotype_data["evaluations"]
        default_evaluation_data = evaluation_data[""]
        fitness = [default_evaluation_data[crit] for crit in OPTIMIZATION_CRITERIA]
    except (KeyError, TypeError, NameError) as e:  # the evaluation may have failed for invalid genotypes (or some other reason)
        fitness = [-1] * len(OPTIMIZATION_CRITERIA)
        print("Error '%s': could not evaluate genotype '%s', returning fitness %s" % (str(e), genotype, fitness))
    return fitness


def frams_crossover(evolution_model : EvolutionModel, individual1, individual2):
    geno1 = evolution_model.config['prefix']+individual1[
        0]  # [0] because we can't (?) have a simple str as a deap genotype/individual, only list of str.
    geno2 = evolution_model.config['prefix']+individual2[
        0]  # [0] because we can't (?) have a simple str as a deap genotype/individual, only list of str.
    try:
        t1 = evolution_model.framsCLI.crossOver(geno1, geno2)
        t1 = t1[len(evolution_model.config['prefix']):]
        t2 = evolution_model.framsCLI.crossOver(geno1, geno2)
        t2= t2[len(evolution_model.config['prefix']):]
        individual1 = creator.Individual((t1,))
        individual2 = creator.Individual((t2,))
    except:
        evolution_model.createCLI()
    return individual1, individual2


def frams_mutate(evolution_model : EvolutionModel, individual):
    try:
        t= evolution_model.framsCLI.mutate(evolution_model.config['prefix']+individual[0])[len(evolution_model.config['prefix']):]
        individual = creator.Individual((t,))
    except:
        evolution_model.createCLI()
    return individual,


def frams_getsimplest(evolution_model : EvolutionModel, genetic_format, genos):
    max = len(genos)
    return extract_fram(genos[np.random.randint(max)])


# def selTournament_keepBest(individuals, k, tournsize, fit_attr="fitness"):
#     """Select the best individual among *tournsize* randomly chosen
#     individuals, *k* times. The list returned contains
#     references to the input *individuals*.
#
#     :param individuals: A list of individuals to select from.
#     :param k: The number of individuals to select.
#     :param tournsize: The number of individuals participating in each tournament.
#     :param fit_attr: The attribute of individuals to use as selection criterion
#     :returns: A list of selected individuals.
#
#     This function uses the :func:`~random.choice` function from the python base
#     :mod:`random` module.
#     """
#     best = max(individuals, key=attrgetter(fit_attr))
#     chosen = []
#     for i in range(k):
#         aspirants = tools.selRandom(individuals, tournsize)
#         chosen.append(max(aspirants, key=attrgetter(fit_attr)))
#     if best not in chosen:
#         chosen.append(best)
#     else:
#         aspirants = tools.selRandom(individuals, tournsize)
#         chosen.append(max(aspirants, key=attrgetter(fit_attr)))
#     return chosen

def prepareToolbox(latent, evolution_model : EvolutionModel, popsize):
    creator.create("FitnessMax", base.Fitness, weights=[1.0] * len(OPTIMIZATION_CRITERIA))
    creator.create("Individual", np.ndarray,
                   fitness=creator.FitnessMax)  # would be nice to have "str" instead of unnecessary "list of str"
    toolbox = base.Toolbox()
    if latent == 'nolatent':
        simplest = frams_getsimplest
        eval = frams_evaluate
        cross = frams_crossover
        mut = frams_mutate
        additional_arg = evolution_model.data[2]
        cli_arg = evolution_model
    else:
        simplest = frams_getsimplest_lat
        eval = frams_evaluate_lat
        cross = frams_crossover_lat2
        mut = frams_mutate_lat
        additional_arg = evolution_model.data[2]
        cli_arg = evolution_model



        if 'onePlus' in evolution_model.config and evolution_model.config['onePlus']:
            parent = creator.Individual((evolution_model.means,))
            parent.fitness.values = [0.00]
            strategy = StrategyOnePlusLambda(parent=parent, sigma=1.0, lambda_=popsize)
        else:
            strategy = Strategy(centroid=evolution_model.means, sigma=1.0, cmatrix=evolution_model.cov,
                                     lambda_=popsize,
                                     mu=int(popsize * 0.25), weights='superlinear')

        toolbox.register("generate", strategy.generate, lambda x: creator.Individual((x,)), evolution_model)
        toolbox.register("update", strategy.update)

    toolbox.register("attr_simplest_genotype", simplest, cli_arg, evolution_model.config['representation'][1:],
                     additional_arg)  # "Attribute generator"
    # (failed) struggle to have an individual which is a simple str, not a list of str
    # toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_frams)
    # https://stackoverflow.com/questions/51451815/python-deap-library-using-random-words-as-individuals
    # https://github.com/DEAP/deap/issues/339
    # https://gitlab.com/santiagoandre/deap-customize-population-example/-/blob/master/AGbasic.py
    # https://groups.google.com/forum/#!topic/deap-users/22g1kyrpKy8
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_simplest_genotype, 1)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", eval, cli_arg)
    toolbox.register("mate", cross, cli_arg)
    toolbox.register("mutate", mut, cli_arg)

    # from deap.creator import Individual

    if len(OPTIMIZATION_CRITERIA) == 1:
        toolbox.register("select", tools.selTournament, tournsize=evolution_model.config['tourney_size'])
    else:
        toolbox.register("select", tools.selNSGA2)
    return toolbox


# def create_config(data_path, load_dir):
#     # representation = 'f1'
#     # cells = 64
#     # long_genos = None
#     # twoLayer = 'oneLayer'
#     # bidir = 'Bidir'
#     #
#     #
#
#
#     data_path = list(os.path.split(data_path))
#     params = data_path[1].split('_')
#     model_name = '_'.join(params[1:-4])
#     representation = params[1]
#     cells =int(params[3])
#     long_genos = None
#     twoLayer = params[4]
#     bidir = params[5]
#     loc = params[6]
#     tes = params[7]
#     model_name = 'model_' + representation + '_' + str(long_genos) + '_' + str(cells) + '_' + twoLayer + '_' + bidir + '_' + loc + '_' + tes
#     print(model_name)
#     data_path = os.path.join(data_path[0], model_name)
#     max_len = 100
#     config = get_config(model_name, representation, long_genos, cells, twoLayer, bidir, data_path, load_dir,
#                         max_len=max_len, locality=loc, test=tes)
#     # config['features'] = 21
#
#     return config


def multiple_evaluate_lat(evolution_model:EvolutionModel, individuals):
    individuals = np.array([individual[0] for individual in individuals])
    frams = evolution_model.decode_latents(individuals)
    # print(frams)
    fitnesses = evaluate_with_batches(evolution_model, frams, base_fitness=-1)
    # print(fitnesses)
    return fitnesses

def multiple_evaluate_frams(evolution_model:EvolutionModel, individuals):
    return evaluate_with_batches(evolution_model, [ind[0] for ind in individuals], base_fitness=-1)

import copy
from math import sqrt, log, exp
import numpy
class StrategyOnePlusLambda(object):
    """
    A CMA-ES strategy that uses the :math:`1 + \lambda` paradigm ([Igel2007]_).

    :param parent: An iterable object that indicates where to start the
                   evolution. The parent requires a fitness attribute.
    :param sigma: The initial standard deviation of the distribution.
    :param lambda_: Number of offspring to produce from the parent.
                    (optional, defaults to 1)
    :param parameter: One or more parameter to pass to the strategy as
                      described in the following table. (optional)

    Other parameters can be provided as described in the next table

    +----------------+---------------------------+----------------------------+
    | Parameter      | Default                   | Details                    |
    +================+===========================+============================+
    | ``d``          | ``1.0 + N / (2.0 *        | Damping for step-size.     |
    |                | lambda_)``                |                            |
    +----------------+---------------------------+----------------------------+
    | ``ptarg``      | ``1.0 / (5 + sqrt(lambda_)| Taget success rate.        |
    |                | / 2.0)``                  |                            |
    +----------------+---------------------------+----------------------------+
    | ``cp``         | ``ptarg * lambda_ / (2.0 +| Step size learning rate.   |
    |                | ptarg * lambda_)``        |                            |
    +----------------+---------------------------+----------------------------+
    | ``cc``         | ``2.0 / (N + 2.0)``       | Cumulation time horizon.   |
    +----------------+---------------------------+----------------------------+
    | ``ccov``       | ``2.0 / (N**2 + 6.0)``    | Covariance matrix learning |
    |                |                           | rate.                      |
    +----------------+---------------------------+----------------------------+
    | ``pthresh``    | ``0.44``                  | Threshold success rate.    |
    +----------------+---------------------------+----------------------------+

    .. [Igel2007] Igel, Hansen, Roth, 2007. Covariance matrix adaptation for
    multi-objective optimization. *Evolutionary Computation* Spring;15(1):1-28

    """
    def __init__(self, parent, sigma, **kargs):
        self.parent = parent
        self.sigma = sigma
        self.dim = len(self.parent[0])

        self.C = numpy.identity(self.dim)
        self.A = numpy.identity(self.dim)

        self.pc = numpy.zeros(self.dim)

        self.computeParams(kargs)
        self.psucc = self.ptarg

    def computeParams(self, params):
        """Computes the parameters depending on :math:`\lambda`. It needs to
        be called again if :math:`\lambda` changes during evolution.

        :param params: A dictionary of the manually set parameters.
        """
        # Selection :
        self.lambda_ = params.get("lambda_", 1)

        # Step size control :
        self.d = params.get("d", 1.0 + self.dim / (2.0 * self.lambda_))
        self.ptarg = params.get("ptarg", 1.0 / (5 + sqrt(self.lambda_) / 2.0))
        self.cp = params.get("cp", self.ptarg * self.lambda_ / (2 + self.ptarg * self.lambda_))

        # Covariance matrix adaptation
        self.cc = params.get("cc", 2.0 / (self.dim + 2.0))
        self.ccov = params.get("ccov", 2.0 / (self.dim ** 2 + 6.0))
        self.pthresh = params.get("pthresh", 0.44)

    def gen_valid(self, howMuch, evolution_model : EvolutionModel, check_valid = False):
        if check_valid:
            max_tests = 200

            arz = numpy.random.standard_normal((max_tests, self.dim))
            arz = self.parent[0] + self.sigma * numpy.dot(arz, self.A.T)
            # arz[:64] = np.clip(arz, a_min=-1.0, a_max=1.0)
            frams = evolution_model.decode_latents(arz)

            frams = [evolution_model.config['prefix'] + fram for fram in frams]
            try:
                valids = evolution_model.framsCLI.isValid(frams)
                # valids = [valid for valid in valids]
                indices = np.where(np.array(valids))[0]
                np.random.shuffle(indices)
                indices = indices[:howMuch]
                latents = arz[indices]
                if len(indices) < howMuch:
                    latents2 = self.gen_valid(howMuch-len(indices), evolution_model, check_valid)
                    latents = np.concatenate([latents, latents2], axis=0)



            except:
                evolution_model.createCLI()
                latents = self.gen_valid(howMuch, evolution_model, check_valid)
        else:
            arz = numpy.random.standard_normal((howMuch, self.dim))
            latents = self.parent[0] + self.sigma * numpy.dot(arz, self.A.T)
        return latents

    def generate(self, ind_init, evolution_model : EvolutionModel):
        """Generate a population of :math:`\lambda` individuals of type
        *ind_init* from the current strategy.

        :param ind_init: A function object that is able to initialize an
                         individual from a list.
        :returns: A list of individuals.
        """

        arz = self.gen_valid(self.lambda_, evolution_model, check_valid=evolution_model.config['check_valid'])




        return list(map(ind_init, arz))


    def update(self, population):
        """Update the current covariance matrix strategy from the
        *population*.

        :param population: A list of individuals from which to update the
                           parameters.
        """
        population.sort(key=lambda ind: ind.fitness, reverse=True)
        lambda_succ = sum(self.parent.fitness <= ind.fitness for ind in population)
        p_succ = float(lambda_succ) / self.lambda_
        self.psucc = (1 - self.cp) * self.psucc + self.cp * p_succ

        if self.parent.fitness <= population[0].fitness:
            x_step = (population[0] - numpy.array(self.parent[0])) / self.sigma
            self.parent = copy.deepcopy(population[0])
            if self.psucc < self.pthresh:
                self.pc = (1 - self.cc) * self.pc + sqrt(self.cc * (2 - self.cc)) * x_step
                self.C = (1 - self.ccov) * self.C + self.ccov * numpy.outer(self.pc, self.pc)
            else:
                self.pc = (1 - self.cc) * self.pc
                self.C = (1 - self.ccov) * self.C + self.ccov * (numpy.outer(self.pc, self.pc) + self.cc * (2 - self.cc) * self.C)

        self.sigma = self.sigma * exp(1.0 / self.d * (self.psucc - self.ptarg) / (1.0 - self.ptarg))

        # We use Cholesky since for now we have no use of eigen decomposition
        # Basically, Cholesky returns a matrix A as C = A*A.T
        # Eigen decomposition returns two matrix B and D^2 as C = B*D^2*B.T = B*D*D*B.T
        # So A == B*D
        # To compute the new individual we need to multiply each vector z by A
        # as y = centroid + sigma * A*z
        # So the Cholesky is more straightforward as we don't need to compute
        # the squareroot of D^2, and multiply B and D in order to get A, we directly get A.
        # This can't be done (without cost) with the standard CMA-ES as the eigen decomposition is used
        # to compute covariance matrix inverse in the step-size evolutionary path computation.
        self.A = numpy.linalg.cholesky(self.C)

from deap.cma import Strategy
class Strategy(object):
    """
    A strategy that will keep track of the basic parameters of the CMA-ES
    algorithm ([Hansen2001]_).

    :param centroid: An iterable object that indicates where to start the
                     evolution.
    :param sigma: The initial standard deviation of the distribution.
    :param parameter: One or more parameter to pass to the strategy as
                      described in the following table, optional.

    +----------------+---------------------------+----------------------------+
    | Parameter      | Default                   | Details                    |
    +================+===========================+============================+
    | ``lambda_``    | ``int(4 + 3 * log(N))``   | Number of children to      |
    |                |                           | produce at each generation,|
    |                |                           | ``N`` is the individual's  |
    |                |                           | size (integer).            |
    +----------------+---------------------------+----------------------------+
    | ``mu``         | ``int(lambda_ / 2)``      | The number of parents to   |
    |                |                           | keep from the              |
    |                |                           | lambda children (integer). |
    +----------------+---------------------------+----------------------------+
    | ``cmatrix``    | ``identity(N)``           | The initial covariance     |
    |                |                           | matrix of the distribution |
    |                |                           | that will be sampled.      |
    +----------------+---------------------------+----------------------------+
    | ``weights``    | ``"superlinear"``         | Decrease speed, can be     |
    |                |                           | ``"superlinear"``,         |
    |                |                           | ``"linear"`` or            |
    |                |                           | ``"equal"``.               |
    +----------------+---------------------------+----------------------------+
    | ``cs``         | ``(mueff + 2) /           | Cumulation constant for    |
    |                | (N + mueff + 3)``         | step-size.                 |
    +----------------+---------------------------+----------------------------+
    | ``damps``      | ``1 + 2 * max(0, sqrt((   | Damping for step-size.     |
    |                | mueff - 1) / (N + 1)) - 1)|                            |
    |                | + cs``                    |                            |
    +----------------+---------------------------+----------------------------+
    | ``ccum``       | ``4 / (N + 4)``           | Cumulation constant for    |
    |                |                           | covariance matrix.         |
    +----------------+---------------------------+----------------------------+
    | ``ccov1``      | ``2 / ((N + 1.3)^2 +      | Learning rate for rank-one |
    |                | mueff)``                  | update.                    |
    +----------------+---------------------------+----------------------------+
    | ``ccovmu``     | ``2 * (mueff - 2 + 1 /    | Learning rate for rank-mu  |
    |                | mueff) / ((N + 2)^2 +     | update.                    |
    |                | mueff)``                  |                            |
    +----------------+---------------------------+----------------------------+

    .. [Hansen2001] Hansen and Ostermeier, 2001. Completely Derandomized
       Self-Adaptation in Evolution Strategies. *Evolutionary Computation*

    """
    def __init__(self, centroid, sigma, **kargs):
        self.params = kargs

        # Create a centroid as a numpy array
        self.centroid = numpy.array(centroid)

        self.dim = len(self.centroid)
        self.sigma = sigma
        self.pc = numpy.zeros(self.dim)
        self.ps = numpy.zeros(self.dim)
        self.chiN = sqrt(self.dim) * (1 - 1. / (4. * self.dim) +
                                      1. / (21. * self.dim ** 2))

        self.C = self.params.get("cmatrix", numpy.identity(self.dim))
        self.diagD, self.B = numpy.linalg.eigh(self.C)

        indx = numpy.argsort(self.diagD)
        self.diagD = self.diagD[indx] ** 0.5
        self.B = self.B[:, indx]
        self.BD = self.B * self.diagD

        self.cond = self.diagD[indx[-1]] / self.diagD[indx[0]]

        self.lambda_ = self.params.get("lambda_", int(4 + 3 * log(self.dim)))
        self.update_count = 0
        self.computeParams(self.params)

    def gen_valid(self, howMuch, evolution_model : EvolutionModel, check_valid = False):
        if check_valid:
            max_tests = 200

            arz = numpy.random.standard_normal((max_tests, self.dim))
            arz = self.centroid + self.sigma * numpy.dot(arz, self.BD.T)
            # arz[:64] = np.clip(arz, a_min=-1.0, a_max=1.0)
            frams = evolution_model.decode_latents(arz)

            frams = [evolution_model.config['prefix'] + fram for fram in frams]
            try:
                valids = evolution_model.framsCLI.isValid(frams)
                # valids = [valid for valid in valids]
                indices = np.where(np.array(valids))[0]
                np.random.shuffle(indices)
                indices = indices[:howMuch]
                latents = arz[indices]
                if len(indices) < howMuch:
                    latents2 = self.gen_valid(howMuch-len(indices), evolution_model, check_valid)
                    latents = np.concatenate([latents, latents2], axis=0)



            except:
                evolution_model.createCLI()
                latents = self.gen_valid(howMuch, evolution_model, check_valid)
        else:
            arz = numpy.random.standard_normal((howMuch, self.dim))
            latents = self.centroid + self.sigma * numpy.dot(arz, self.BD.T)
        return latents



    def generate(self, ind_init, evolution_model : EvolutionModel):
        """Generate a population of :math:`\lambda` individuals of type
        *ind_init* from the current strategy.

        :param ind_init: A function object that is able to initialize an
                         individual from a list.
        :returns: A list of individuals.
        """

        arz = self.gen_valid(self.lambda_, evolution_model, check_valid=evolution_model.config['check_valid'])




        return list(map(ind_init, arz))

    def update(self, population):
        """Update the current covariance matrix strategy from the
        *population*.

        :param population: A list of individuals from which to update the
                           parameters.
        """
        population.sort(key=lambda ind: ind.fitness, reverse=True)
        population = [individual[0] for individual in population]
        old_centroid = self.centroid
        self.centroid = numpy.dot(self.weights, population[0:self.mu])

        c_diff = self.centroid - old_centroid

        # Cumulation : update evolution path
        self.ps = (1 - self.cs) * self.ps \
            + sqrt(self.cs * (2 - self.cs) * self.mueff) / self.sigma \
            * numpy.dot(self.B, (1. / self.diagD) *
                        numpy.dot(self.B.T, c_diff))

        hsig = float((numpy.linalg.norm(self.ps) /
                      sqrt(1. - (1. - self.cs) ** (2. * (self.update_count + 1.))) / self.chiN <
                      (1.4 + 2. / (self.dim + 1.))))

        self.update_count += 1

        self.pc = (1 - self.cc) * self.pc + hsig \
            * sqrt(self.cc * (2 - self.cc) * self.mueff) / self.sigma \
            * c_diff

        # Update covariance matrix
        artmp = population[0:self.mu] - old_centroid
        self.C = (1 - self.ccov1 - self.ccovmu + (1 - hsig) *
                  self.ccov1 * self.cc * (2 - self.cc)) * self.C \
            + self.ccov1 * numpy.outer(self.pc, self.pc) \
            + self.ccovmu * numpy.dot((self.weights * artmp.T), artmp) \
            / self.sigma ** 2

        self.sigma *= numpy.exp((numpy.linalg.norm(self.ps) / self.chiN - 1.) *
                                self.cs / self.damps)

        self.diagD, self.B = numpy.linalg.eigh(self.C)
        indx = numpy.argsort(self.diagD)

        self.cond = self.diagD[indx[-1]] / self.diagD[indx[0]]

        self.diagD = self.diagD[indx] ** 0.5
        self.B = self.B[:, indx]
        self.BD = self.B * self.diagD

    def computeParams(self, params):
        """Computes the parameters depending on :math:`\lambda`. It needs to
        be called again if :math:`\lambda` changes during evolution.

        :param params: A dictionary of the manually set parameters.
        """
        self.mu = params.get("mu", int(self.lambda_ / 2))
        rweights = params.get("weights", "superlinear")
        if rweights == "superlinear":
            self.weights = log(self.mu + 0.5) - \
                numpy.log(numpy.arange(1, self.mu + 1))
        elif rweights == "linear":
            self.weights = self.mu + 0.5 - numpy.arange(1, self.mu + 1)
        elif rweights == "equal":
            self.weights = numpy.ones(self.mu)
        else:
            raise RuntimeError("Unknown weights : %s" % rweights)

        self.weights /= sum(self.weights)
        self.mueff = 1. / sum(self.weights ** 2)

        self.cc = params.get("ccum", 4. / (self.dim + 4.))
        self.cs = params.get("cs", (self.mueff + 2.) /
                             (self.dim + self.mueff + 3.))
        self.ccov1 = params.get("ccov1", 2. / ((self.dim + 1.3) ** 2 +
                                               self.mueff))
        self.ccovmu = params.get("ccovmu", 2. * (self.mueff - 2. +
                                                 1. / self.mueff) /
                                 ((self.dim + 2.) ** 2 + self.mueff))
        self.ccovmu = min(1 - self.ccov1, self.ccovmu)
        self.damps = 1. + 2. * max(0, sqrt((self.mueff - 1.) /
                                           (self.dim + 1.)) - 1.) + self.cs
        self.damps = params.get("damps", self.damps)

def eaGenerateUpdate2(toolbox, ngen , evolution_model : EvolutionModel, multiple_evaluate_f, halloffame=None, stats=None,
                     verbose=__debug__):
    """This is algorithm implements the ask-tell model proposed in
    [Colette2010]_, where ask is called `generate` and tell is called `update`.

    :param toolbox: A :class:`~deap.base.Toolbox` that contains the evolution
                    operators.
    :param ngen: The number of generation.
    :param stats: A :class:`~deap.tools.Statistics` object that is updated
                  inplace, optional.
    :param halloffame: A :class:`~deap.tools.HallOfFame` object that will
                       contain the best individuals, optional.
    :param verbose: Whether or not to log the statistics.
    :returns: The final population
    :returns: A class:`~deap.tools.Logbook` with the statistics of the
              evolution

    The algorithm generates the individuals using the :func:`toolbox.generate`
    function and updates the generation method with the :func:`toolbox.update`
    function. It returns the optimized population and a
    :class:`~deap.tools.Logbook` with the statistics of the evolution. The
    logbook will contain the generation number, the number of evaluations for
    each generation and the statistics if a :class:`~deap.tools.Statistics` is
    given as argument. The pseudocode goes as follow ::

        for g in range(ngen):
            population = toolbox.generate()
            evaluate(population)
            toolbox.update(population)

    .. [Colette2010] Collette, Y., N. Hansen, G. Pujol, D. Salazar Aponte and
       R. Le Riche (2010). On Object-Oriented Programming of Optimizers -
       Examples in Scilab. In P. Breitkopf and R. F. Coelho, eds.:
       Multidisciplinary Design Optimization in Computational Mechanics,
       Wiley, pp. 527-565;

    """
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    for gen in range(ngen):
        # Generate a new population
        population = toolbox.generate()
        # Evaluate the individuals
        fitnesses = multiple_evaluate_f(evolution_model, population)
        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = [fit]

        if halloffame is not None:
            halloffame.update(population)

        # Update the strategy with the evaluated individuals
        toolbox.update(population)

        record = stats.compile(population) if stats is not None else {}
        logbook.record(gen=gen, nevals=len(population), **record)
        if verbose:
            print(logbook.stream)

    return population, logbook

def eaSimple2(population, toolbox, cxpb, mutpb, ngen, evolution_model : EvolutionModel, multiple_evaluate_f, stats=None,
             halloffame=None, verbose=__debug__, logbook=None, keepbest=False, save_every=None):
    """This algorithm reproduce the simplest evolutionary algorithm as
    presented in chapter 7 of [Back2000]_.

    :param population: A list of individuals.
    :param toolbox: A :class:`~deap.base.Toolbox` that contains the evolution
                    operators.
    :param cxpb: The probability of mating two individuals.
    :param mutpb: The probability of mutating an individual.
    :param ngen: The number of generation.
    :param stats: A :class:`~deap.tools.Statistics` object that is updated
                  inplace, optional.
    :param halloffame: A :class:`~deap.tools.HallOfFame` object that will
                       contain the best individuals, optional.
    :param verbose: Whether or not to log the statistics.
    :returns: The final population
    :returns: A class:`~deap.tools.Logbook` with the statistics of the
              evolution

    The algorithm takes in a population and evolves it in place using the
    :meth:`varAnd` method. It returns the optimized population and a
    :class:`~deap.tools.Logbook` with the statistics of the evolution. The
    logbook will contain the generation number, the number of evaluations for
    each generation and the statistics if a :class:`~deap.tools.Statistics` is
    given as argument. The *cxpb* and *mutpb* arguments are passed to the
    :func:`varAnd` function. The pseudocode goes as follow ::

        evaluate(population)
        for g in range(ngen):
            population = select(population, len(population))
            offspring = varAnd(population, toolbox, cxpb, mutpb)
            evaluate(offspring)
            population = offspring

    As stated in the pseudocode above, the algorithm goes as follow. First, it
    evaluates the individuals with an invalid fitness. Second, it enters the
    generational loop where the selection procedure is applied to entirely
    replace the parental population. The 1:1 replacement ratio of this
    algorithm **requires** the selection procedure to be stochastic and to
    select multiple times the same individual, for example,
    :func:`~deap.tools.selTournament` and :func:`~deap.tools.selRoulette`.
    Third, it applies the :func:`varAnd` function to produce the next
    generation population. Fourth, it evaluates the new individuals and
    compute the statistics on this population. Finally, when *ngen*
    generations are done, the algorithm returns a tuple with the final
    population and a :class:`~deap.tools.Logbook` of the evolution.

    .. note::

        Using a non-stochastic selection method will result in no selection as
        the operator selects *n* individuals from a pool of *n*.

    This function expects the :meth:`toolbox.mate`, :meth:`toolbox.mutate`,
    :meth:`toolbox.select` and :meth:`toolbox.evaluate` aliases to be
    registered in the toolbox.

    .. [Back2000] Back, Fogel and Michalewicz, "Evolutionary Computation 1 :
       Basic Algorithms and Operators", 2000.
    """
    if logbook is None:
        logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in population if not ind.fitness.valid]



    fitnesses = multiple_evaluate_f(evolution_model, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = [fit]

    if halloffame is not None:
        halloffame.update(population)

    record = stats.compile(population) if stats else {}
    logbook.record(gen=0, nevals=len(invalid_ind), **record)
    if verbose:
        print(logbook.stream, flush=True)

    # Begin the generational process
    for gen in range(1, ngen + 1):
        # Select the next generation individuals
        best = max(population, key=attrgetter("fitness"))

        offspring = toolbox.select(population, len(population))

        # Vary the pool of individuals
        offspring = varAnd(offspring, toolbox, cxpb, mutpb)
        seen = set([str(off) for off in offspring])
        if keepbest and str(best) not in seen:
            offspring.pop(random.randint(0, len(offspring)-1))
            offspring.append(best)
        random.shuffle(offspring)

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = multiple_evaluate_f(evolution_model, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = [fit]

        # Update the hall of fame with the generated individuals
        if halloffame is not None:
            halloffame.update(offspring)

        # Replace the current population by the offspring
        population[:] = offspring

        # Append the current generation statistics to the logbook
        record = stats.compile(population) if stats else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        if verbose:
            print(logbook.stream, flush=True)
        if save_every:
            save_every(logbook)

    return population, logbook


def runEvolLatent(config, gene, pop_s, latent, cmaes=False, iterations=10, redir_out = True):
    max_iterations = 50
    if iterations > max_iterations:
        iterations = max_iterations
    # A demo run: optimize OPTIMIZATION_CRITERIA
    # find max logbook
    iteration = -1
    files = []
    for filename in glob.glob(os.path.join(config['data_path'], "logbook_*")):
        try:
            files.append((filename, int(filename.split('_')[-1])))
        except:
            pass

    files = list(set(files))
    iteration += config['task_test'] * max_iterations
    files = [(filename, ind) for filename, ind in files if ind > iteration and ind <= iteration + max_iterations]
    if len(files) > 0:
        files = sorted(files, key=lambda x: x[1], reverse=True)
        iteration = int(files[0][1])
    # random.seed(123)  # see FramsticksCLI.DETERMINISTIC below, set to True if you want full determinism

    # if config is None:
    #     config = create_config(data_path, load_dir)







    # framsCLI.PRINT_FRAMSTICKS_OUTPUT = True
    #
    # framsCLI.PRINT_FRAMSTICKS_OUTPUT = False
    POPSIZE = pop_s
    GENERATIONS = gene
    config['POPSIZE'] = POPSIZE
    config['GENERATIONS'] = GENERATIONS






    while iteration < max_iterations * (config['task_test']+1)-1:
        iteration += 1
        with open(os.path.join(config['data_path'],'experiment_outFile_%s' % str(iteration)), 'w') as outF:
            if redir_out:
                sys.stdout = outF
            if latent == 'nolatent':
                # config['importSim'].append('Simulator.import(\"generation_params.sim\");')
                # config['markers'].append("Simulator.load")
                multiple_evaluate_f = multiple_evaluate_frams
                EM = EvolutionModel(config, load_autoencoder=False)
            else:
                multiple_evaluate_f = multiple_evaluate_lat
                EM = EvolutionModel(config)

            FramsticksCLI.DETERMINISTIC = False  # must be set before FramsticksCLI() constructor call

            EM.createCLI()

            toolbox = prepareToolbox(latent,
                                     evolution_model=EM, popsize=POPSIZE)



            print("Representation model: " + EM.config['model_name'])
            print("Fake fitness: " + str(fake_fitness[0]))
            print("Fake mutate: " + str(fake_mutate[0]))
            print("Fitness len weight: " + str(fitness_len_weight[0]))
            print("Fitness max len value: " + str(fitness_len_max_value[0]))
            print("Fitness max len: " + str(fitness_max_len[0]))
            print("Fitness len sub char" + str( fitness_len_chars_sub[0]))
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
            sys.stdout.flush()
            def save_results(logbook):
                with open(os.path.join(config['data_path'], 'logbook_%s' % str(iteration)), 'wb') as f:
                    pickle.dump([logbook, EM.config['model_name'], fake_fitness[0], config['evol_use_encoder'], config['evol_keepbest'], fake_mutate[0], config, fitness_len_weight[0], fitness_len_max_value[0], fitness_max_len[0], fitness_len_chars_sub[0], fitness_min_value[0]], f)

            def runAlg(pop, log):
                try:
                    if not cmaes:
                        pop, log = eaSimple2(pop, toolbox, cxpb=0.15, mutpb=0.6, ngen=GENERATIONS,
                                                        evolution_model = EM,
                                                        multiple_evaluate_f=multiple_evaluate_f,
                                                        stats=stats,
                                                       halloffame=hof,
                                                       verbose=True,
                                             keepbest=config['evol_keepbest'], save_every=save_results)
                    else:
                        pop, log = eaGenerateUpdate2(toolbox, ngen=GENERATIONS, stats=stats, halloffame=hof, evolution_model=EM, multiple_evaluate_f=multiple_evaluate_f, verbose=True)
                except Exception as e:
                    print(e)
                    print("Error! Stopping!")
                return pop, log


            pop, logbook = runAlg(pop, logbook)
            save_results(logbook)
            print('Best individuals:')
            for best in hof:
                if latent == 'nolatent':
                    print(best.fitness, '\t-->\t', best[0])
                else:
                    fram = EM.decode_latents(np.expand_dims(best[0], axis=0))[0]
                    print(best.fitness, '\t-->\t', fram)
                    print(best[0])


#


