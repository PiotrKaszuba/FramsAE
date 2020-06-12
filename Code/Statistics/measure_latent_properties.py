import os

import numpy as np
import tensorflow as tf

from Code.Preparation.configuration import get_config
from Code.Preparation.Utils import extract_fram, get_genSuff

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from Levenshtein import distance
from textwrap import wrap
from Code.FramsticksCLI import FramsticksCLI
from collections import Counter

from Code.Preparation.Run_preparation import prepareData, EvolutionModel, getConcatEncoderPredGenotypes, splitLatentToHiddenAndCell


####################################

def get_title(title, line_len=60):
    return "\n".join(wrap(title, line_len))



def get_Accs(genos, howMany, framsCLI, config):
    lengs = Counter()
    lengsHit = Counter()

    hit = 0
    for g in range(howMany):
        fram = extract_fram(genos[g])  # re.sub('[ST]', '', fram)
        print(g, fram)
        isValid = framsCLI.isValid(str(get_genSuff(config['representation'])) + str(fram))
        hit += isValid
        lengsHit[len(fram)] += isValid
        lengs[len(fram)] += 1
        print(isValid)
        print("-------------")

    xs = []
    ys = []
    accs = []
    for i in range(config['max_len']):
        if lengs[i] > 0:
            accs.append(lengsHit[i] / lengs[i])
            xs.append(i)
            ys.append(lengs[i])
    total_acc = hit / howMany

    return total_acc, xs, accs, ys


def plot_validitiy(xs, accs, ys, mutations, howMany, total_acc, sett, suff, config):
    fig, ax1 = plt.subplots()
    fig.set_figwidth(12.0)
    fig.set_figheight(7.0)
    color = 'tab:blue'
    ax1.set_xlabel('Geno length')
    ax1.set_ylabel('% validity over sample', color=color)
    ax1.plot(xs, accs, color=color)
    ax1.scatter(xs, accs, color=color, s=1.5)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:red'
    ax2.set_ylabel('sample size', color=color)  # we already handled the x-label with ax1
    ax2.plot(xs, ys, color=color)
    ax2.scatter(xs, ys, color=color, s=1.5)
    ax2.tick_params(axis='y', labelcolor=color)

    # fig.tight_layout()  # otherwise the right y-label is slightly clipped

    ax1.xaxis.grid()  # vertical lines
    # ax1.set_xticks(np.arange(0, config['max_len'] + 1, 5))
    mean_length = np.average(xs, weights=ys)

    fig.suptitle(get_title(
        config['model_name'] + ', mutations: ' + str(mutations) + ', sample: ' + str(howMany) + ', total validity %: ' + str(
            total_acc) + ', mean length: ' + str(mean_length) + ', set: ' + str(sett) + ' ' + str(suff)))

    plt.savefig(
        os.path.join("plots/", config['model_name'] + '_' + str(sett) + '_' + str(mutations) + str(suff) + '_validity_plot.png'))
    plt.clf()


def checkValidity(genos, howMany, config):
    framsCLI = FramsticksCLI('C:/Users/Piotr/Desktop/Framsticks50rc14', None)

    total_acc, xs, accs, ys = get_Accs(genos, howMany, framsCLI, config)

    framsCLI.closeFramsticksCLI()

    return total_acc, xs, accs, ys


def validity_of_sampled(enc_predict_train, autoencoder_model):
    min1 = np.min(enc_predict_train[0], axis=0)
    min2 = np.min(enc_predict_train[1], axis=0)
    max1 = np.max(enc_predict_train[0], axis=0)
    max2 = np.max(enc_predict_train[1], axis=0)

    q12 = np.quantile(enc_predict_train[1], 0.1, axis=0)
    q32 = np.quantile(enc_predict_train[1], 0.9, axis=0)




    howMany = 100
    uni1 = np.random.uniform(min1, max1, size=(howMany, autoencoder_model.config['cells']))
    uni2 = np.random.uniform(min2, max2, size=(howMany, autoencoder_model.config['cells']))
    decOnes = autoencoder_model.getOnesMaskForFullPrediction(howMany)

    pred = autoencoder_model.decoder.predict([uni1, uni2, decOnes])
    genos = autoencoder_model.get_genos(pred)

    total_acc, xs, accs, ys = checkValidity(genos, howMany, autoencoder_model.config)
    plot_validitiy(xs, accs, ys, '', howMany, total_acc, 'UNIFORM_SAMPLE_FULL', suff='', config=autoencoder_model.config)
    #######################################################################
    uni1 = np.random.uniform(min1, max1, size=(howMany, autoencoder_model.config['cells']))
    uni2 = np.random.uniform(q12, q32, size=(howMany, autoencoder_model.config['cells']))

    pred = autoencoder_model.decoder.predict([uni1, uni2, decOnes])
    genos = autoencoder_model.get_genos(pred)

    total_acc, xs, accs, ys = checkValidity(genos, howMany, autoencoder_model.config)
    plot_validitiy(xs, accs, ys, '', howMany, total_acc, 'UNIFORM_SAMPLE_INTERQUARTILE', suff='', config=autoencoder_model.config)
    ########################################################################


    gaus = autoencoder_model.sampleMultivariateGaussianLatent(howMany)
    hidden, cell = splitLatentToHiddenAndCell(gaus)

    pred = autoencoder_model.decoder.predict([hidden, cell, decOnes])
    genos = autoencoder_model.get_genos(pred)

    total_acc, xs, accs, ys = checkValidity(genos, howMany, autoencoder_model.config)
    plot_validitiy(xs, accs, ys, '', howMany, total_acc, 'MULTIVARIATE_GAUSSIAN_SAMPLE', suff='', config=autoencoder_model.config)
    #############################################################################################


def make_plots(enc_predict, sett, autoencoder_model):
    plt.scatter(np.array([[i] * len(enc_predict[0]) for i in range(autoencoder_model.config['cells'])]).flatten(), enc_predict[0].flatten(), s=0.02)
    plt.title(get_title(autoencoder_model.config['model_name'] + ', latent (hidden state) activations, set: ' + str(sett)))
    plt.xlabel('latent hidden state neuron')
    plt.ylabel('activation')
    plt.savefig(os.path.join("plots/", autoencoder_model.config['model_name'] + '_' + str(sett) + '_activations_hidden.png'))
    plt.clf()
    #
    plt.scatter(np.array([[i] * len(enc_predict[0]) for i in range(autoencoder_model.config['cells'])]).flatten(), enc_predict[1].flatten(), s=0.02)
    plt.title(get_title(autoencoder_model.config['model_name'] + ', latent (cell state) activations, set: ' + str(sett)))
    plt.xlabel('latent cell state neuron')
    plt.ylabel('activation')
    plt.savefig(os.path.join("plots/", autoencoder_model.config['model_name'] + '_' + str(sett) + '_activations_cell.png'))
    plt.clf()

    stds = np.std(enc_predict[0], axis=0)

    plt.bar(np.arange(autoencoder_model.config['cells']), stds)
    plt.title(get_title(autoencoder_model.config['model_name'] + ', latent (hidden state) stds, set: ' + str(sett)))
    plt.xlabel('latent hidden state neuron')
    plt.ylabel('stdev')
    plt.savefig(os.path.join("plots/", autoencoder_model.config['model_name'] + '_' + str(sett) + '_stds_hidden.png'))
    plt.clf()

    stds2 = np.std(enc_predict[1], axis=0)

    plt.bar(np.arange(autoencoder_model.config['cells']), stds2)
    plt.title(get_title(autoencoder_model.config['model_name'] + ', latent (cell state) stds, set: ' + str(sett)))
    plt.xlabel('latent cell state neuron')
    plt.ylabel('stdev')
    plt.savefig(os.path.join("plots/", autoencoder_model.config['model_name'] + '_' + str(sett) + '_stds_cell.png'))
    plt.clf()





def validityMeasurement(enc_latent, enc_latent_train, autoencoder_model, mutations=0, suff=''):
    hidden, cell = splitLatentToHiddenAndCell(enc_latent)
    enc_predict = [hidden, cell, autoencoder_model.getOnesMaskForFullPrediction(np.shape(enc_latent)[0])]

    hidden, cell = splitLatentToHiddenAndCell(enc_latent_train)
    enc_predict_train = [hidden, cell, autoencoder_model.getOnesMaskForFullPrediction(np.shape(enc_latent_train)[0])]

    decTest = autoencoder_model.decoder.predict(enc_predict)
    decTrain = autoencoder_model.decoder.predict(enc_predict_train)
    genosTest = autoencoder_model.get_genos(decTest)
    genosTrain = autoencoder_model.get_genos(decTrain)

    np.random.shuffle(genosTest)
    np.random.shuffle(genosTrain)
    howMany = 100

    total_acc, xs, accs, ys = checkValidity(genosTest, howMany, autoencoder_model.config)

    plot_validitiy(xs, accs, ys, mutations, howMany, total_acc, 'TEST', suff=suff, config=autoencoder_model.config)

    # for g in range(50):
    #
    #     print("-----original-------")
    #     print(np.delete(gen2Check[g], np.where(zerosTestCheck[g] == 0)))
    #     print("-----predicted------")
    #     print(np.delete(genosCheck[g], np.where(zerosTestCheck[g] == 0)))
    #     print("----------prediction full-----------------")
    #     print(genosCheck[g])


def measure_mutation_power_similarity(enc_latent, autoencoder_model, times=20, sett="TEST", suff=''):
    mutation_power_formula = lambda i: (i + 1) * 0.005

    mutated_multiple = []
    howMany = 1000

    inds = np.random.choice(np.shape(enc_latent)[0], howMany, replace=False)

    base_genos = [extract_fram(fram) if ind in inds else fram for ind, fram in
                  enumerate(autoencoder_model.get_genos(autoencoder_model.decoder.predict(list(splitLatentToHiddenAndCell(enc_latent))+[autoencoder_model.getOnesMaskForFullPrediction(np.shape(enc_latent)[0])])))]

    for i in range(times):
        print("mutating, inferring, decoding, extracting genos: " + str(i + 1) + "/" + str(times))
        genos = [extract_fram(fram) if ind in inds else fram for ind, fram in enumerate(
            autoencoder_model.get_genos(autoencoder_model.decoder.predict(list(splitLatentToHiddenAndCell(autoencoder_model.mutate(enc_latent, times=1, magnitude=mutation_power_formula(i))))+[autoencoder_model.getOnesMaskForFullPrediction(np.shape(enc_latent)[0])])))]
        mutated_multiple.append(genos)

    data = []
    for i in range(len(mutated_multiple)):
        data.append((Counter(), Counter()))

    powers = []
    lengs = []
    avg_dists = []
    total_dist = 0
    for i in range(len(mutated_multiple)):
        print("calculating distances for mutation: " + str(i + 1) + "/" + str(times))
        for ind in inds:
            fram = mutated_multiple[i][ind]
            fram_base = base_genos[ind]
            data[i][0][len(fram_base)] += distance(fram, fram_base)
            data[i][1][len(fram_base)] += 1

        all_dist = 0
        for l in range(autoencoder_model.config['max_len'] + 1):
            if data[i][1][l] > 0:
                powers.append(mutation_power_formula(i))
                lengs.append(l)
                avg_dists.append(data[i][0][l] / data[i][1][l])
                all_dist += data[i][0][l]
        total_dist += all_dist / howMany
        print("Average dist for mutation power " + str(mutation_power_formula(i)) + " = " + str(all_dist / howMany))

    avg_total = total_dist / times
    print("Average dist for all mutations powers: " + str(avg_total))

    fig = plt.figure()
    ax1 = fig.gca(projection='3d')
    fig.set_figwidth(12.0)
    fig.set_figheight(7.0)

    ax1.set_xlabel('Geno length')
    ax1.set_ylabel('Mutation power')
    ax1.set_zlabel('Avg edit distance')
    ax1.scatter(lengs, powers, avg_dists, s=2.0)

    maxZ = 50
    maxX = 65
    maxY = mutation_power_formula(times + 1)
    ax1.set_xticks(np.arange(0, maxX + 1, 5))
    ax1.set_zticks(np.arange(0, maxZ + 1, 5))
    ax1.set_yticks(np.arange(0, maxY, mutation_power_formula(0) * 2))

    ax1.set_zlim(0, maxZ)
    ax1.set_xlim(0, maxX)
    ax1.set_ylim(0, maxY)
    ax1.xaxis.grid()  # vertical lines
    ax1.yaxis.grid()
    fig.suptitle(get_title(
        autoencoder_model.config['model_name'] + ', sample: ' + str(howMany * times) + ', total avg distance: ' + str(
            avg_total) + ', set: ' + str(sett) + ' ' + str(suff)))

    plt.show()
    plt.savefig(
        os.path.join("plots/", autoencoder_model.config['model_name'] + '_' + str(sett) + '_' + str(suff) + '_mutation_editdist_plot.png'))
    plt.clf()


def create_config():
    representation = 'f1'
    cells = 64
    long_genos = None
    twoLayer = 'oneLayer'
    bidir = 'Bidir'

    load_dir = 'newGenos'

    model_name = 'model_' + representation + '_' + str(long_genos) + '_' + str(cells) + '_' + twoLayer + '_' + bidir

    model_path = os.path.join('dataFolder', model_name)
    max_len = 100
    config = get_config(model_name, representation, long_genos, cells, twoLayer, bidir, model_path, load_dir,
                        max_len=max_len)
    # config['features'] = 21

    return config


config = create_config()

X_train, Y_train, genosCheck_train = prepareData(config)

# load test genos
temp = config['long_genos']
config['long_genos'] = 'TEST'
X_test, Y_test, genosCheck_test = prepareData(config)
config['long_genos'] = temp

EM = EvolutionModel(config)

print("Predicting..")

enc_predict_train = EM.encoder.predict(X_train)
enc_predict_test = EM.encoder.predict(X_test)

enc_latent_train = getConcatEncoderPredGenotypes(enc_predict_train)
enc_latent_test = getConcatEncoderPredGenotypes(enc_predict_test)


print("Creating plots..")

# make_plots(enc_predict_test, 'TEST', EM)
# make_plots(enc_predict_train, 'TRAIN', EM)


print("Validity of sampled..")

validity_of_sampled(enc_predict_train, EM)

print("mutating..")
enc_predict_mut1 = EM.mutate(enc_latent_test, times=1)
enc_predict_mut1_train = EM.mutate(enc_latent_train, times=1)

enc_predict_mut10 = EM.mutate(enc_latent_test, times=10)
enc_predict_mut10_train = EM.mutate(enc_latent_train, times=10)

print("checking mutations validity..")

validityMeasurement(enc_latent_test, enc_latent_train, EM, mutations=0)
#
validityMeasurement(enc_predict_mut1, enc_predict_mut1_train, EM, mutations=1, suff='_full_cov')

validityMeasurement(enc_predict_mut10, enc_predict_mut10_train, EM, mutations=10, suff='_full_cov')


measure_mutation_power_similarity(enc_latent_test , EM,  times=20)
