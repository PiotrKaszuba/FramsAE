import numpy as np
import pandas as pd

import glob
import os
import pickle

from collections import defaultdict
from scipy.stats import spearmanr
import re

def collectFDCCorrs(data_folder):
    # data_folder, _ = os.path.split(data_folder)
    dataSave = os.path.join(data_folder, 'corrsFile')
    path_base1 = os.path.join(data_folder, "mutateDistance")
    # path_base1 = "mods2/mutateDistance/"
    path_base2 = "**/*corrs"
    path_base = os.path.join(path_base1,path_base2)

    print(path_base, flush=True)
    data = []
    for f in glob.glob(path_base, recursive=True):
        with open(f, 'rb') as file:
            corr, corr_enc = pickle.load(file)

        dirr, filename = os.path.split(f)
        dirr, model_name = os.path.split(dirr)
        data.append((model_name, corr, corr_enc))
    with open(dataSave, 'wb') as file:
        pickle.dump(data, file)
def drawFDCs():
    with open('mods2/corrsFile', 'rb') as file:
        data = pickle.load(file)
    # filter
    cells = '64'
    repr = 'f1'
    data = [(name, cr, cr_enc) for name, cr, cr_enc in data if repr in name]
    import matplotlib.pyplot as plt
    import seaborn as sns
    from mpl_toolkits.mplot3d.axes3d import Axes3D
    from sklearn import preprocessing
    import matplotlib.transforms as transforms
    corrs = {}
    for name, cr, cr_enc in data:
        loc = name.split("_")[6]
        cells = name.split("_")[3]
        if 'f' in loc:
            nam = 'fitness '
        elif 'n' in loc:
            nam = 'standard '
        else:
            nam = 'phenotype '
        nam = str(cells) + " " + nam
        corrs[nam] = cr
        corrs[nam + 'sp'] = cr_enc

    fig = plt.figure(figsize=(6.4, 4.8))
    ax = fig.add_subplot(111)  # , projection='3d')
    vals = np.arange(len(corrs))
    corr_values_points = [cr + 0.05 for cr in corrs.values()]
    points = list(zip(list(vals - 0.25), corr_values_points))
    names = list(corrs.keys())
    names = [re.sub(' +', ' ', nam.replace('model', '')) for nam in names]
    for i in range(len(names)):
        plt.annotate(names[i], points[i])
    ax.scatter(vals, list(corrs.values()), s=30)
    # plt.title('Finess distance correlation for %s models' % repr, fontsize=12)
    plt.ylabel("Fitness distance correlation", fontsize=12, labelpad=10)
    plt.xlabel("%s models with different numbers of LSTM cells" % repr, fontsize=12, labelpad=10)
    plt.tick_params(
        axis='x',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
        labelbottom=False)
    plt.xlim(-1, vals[-1] + 1)
    plt.ylim(-1, 1)
    plt.savefig(
        os.path.join('mods2', 'corrsPlot'+repr),
        dpi=300,
        transparent=False, bbox_inches='tight')
    plt.show()
    plt.clf()


def runFDCforModel(name, data_folder):
    data_folder, _ = os.path.split(data_folder)
    path_base1 = os.path.join(data_folder, "mutateDistance")
    # path_base1 = "mods2/mutateDistance/"
    path_base2 = "**/*mutDist*"
    path_base = os.path.join(path_base1,path_base2)
    repr = name.split('_')[1]

    print(path_base, flush=True)
    def gatherScores(name=None):
        df = pd.DataFrame()

        for f in glob.glob(path_base, recursive=True):
            with open(f, 'rb') as file:
                df_temp = pd.DataFrame(pickle.load(file))

            dirr, filename = os.path.split(f)
            dirr, model_name = os.path.split(dirr)
            if name is not None and name != model_name:
                continue
            if not filename.startswith("model"):

                model_params = model_name.split('_')
            else:
                model_params = filename.split('_')[:-2]
            if model_params[0] == 'model':
                df_temp['representation_base'] = model_params[1]
                df_temp['cells'] = model_params[3]
                df_temp['layers'] = model_params[4]
                df_temp['bidir'] = model_params[5]
                locality_name = model_params[6]
                if locality_name[-1] == 'n':
                    locality = 'noloc'
                if locality_name[-1] == 'l':
                    locality = 'levenshtein'
                if locality_name[-1] == 'f':
                    locality = 'fitness'
                if locality_name[0] == '0' and '-' not in locality_name:
                    locality = 'noloc'
                    learning = 'standard'
                else:
                    learning = 'loc_like_batch'
                    if locality_name[-1] != 'l' and locality_name[-1] != 'n' and locality_name[-1] != 'f':
                        locality = 'dissim'

                if locality_name[-1] == 'l' or locality_name[-1] == 'n' or locality_name[-1] == 'f':
                    locality_power_str = locality_name[:-1]
                else:
                    locality_power_str = locality_name
                locality_power_str = locality_power_str.replace('-', '.')
                print(locality_power_str)
                df_temp['locality'] = locality
                df_temp['learning'] = learning
                df_temp['locality_power'] = float(locality_power_str)
                df_temp['version'] = model_params[7]
            else:
                df_temp['representation_base'] = f.split('_')[0]
            df_temp['name'] = f

            if not filename.startswith("model"):
                df_temp['locality'] = 'orig'
            print(f)
            if df_temp['representation_base'].iloc[0] != repr:
                continue
            df = pd.concat([df, df_temp])
        df.reset_index(drop=True, inplace=True)
        return df


    df = gatherScores(name)
    # from Code.drawDistributions import plot_f, c1


    def FDC(df, model_name=None):
        latents = defaultdict(lambda: None)
        latents_encoder = defaultdict(lambda: None)
        fitnesses = defaultdict(list)
        fitnesses2 = defaultdict(list)
        mutants = defaultdict(list)
        mutants_decoder = defaultdict(list)  # dec
        df = df[df['fitnesses'].notna()]
        # grp = df.groupby('name')
        # print(grp['fitnesses'].mean())
        # df = df[df['mutants_decoder'].notna()]
        maxlen = 20

        # df = df[df['test'] == 16] # TEST FILTERING
        df = df[df['representation_base'] == repr]
        # df = df[df['fake_fitness'] == False]  # filter fake fitness
        names = set()
        for ind, row in df.iterrows():
            names.add(row['name'])
            loc = row['locality']
            # if len(mutants[loc]) > 3000:
            #     continue
            fitnesses[loc].extend(row['fitnesses'])
            fitnesses2[loc].extend(row['fitnesses'])
            mutants[loc].extend(row['mutants'])

            mutants_decoder[loc].extend(row['mutants'])  ### dec
            if loc != 'orig':
                if latents[loc] is None:
                    latents[loc] = row['latents']
                else:
                    latents[loc] = np.concatenate([latents[loc], row['latents']], axis=0)

                if latents_encoder[loc] is None:
                    latents_encoder[loc] = row['latents_encoder']
                else:
                    latents_encoder[loc] = np.concatenate([latents_encoder[loc], row['latents_encoder']], axis=0)
        namemap = {'noloc': 'standard model', 'levenshtein': 'levenshtein', 'dissim': 'phenotype model',
                   'fitness': 'fitness model'}
        corrs = {}
        corrs_enc = {}
        print(names)
        df2 = pd.DataFrame()
        for key in fitnesses.keys():
            fit = fitnesses[key]
            lf = len(fit)
            fit = [f for f in fit if f > -1]
            print(len(fit) / lf)
            print(key, len(fit), np.mean(fit), np.median(fit), np.max(fit))
            df_temp = pd.DataFrame(pd.Series(fit, name='fit'))

            df_temp['name'] = key
            df2 = pd.concat([df2, df_temp])
        # sns.violinplot(x=df2['name'], y=df2['fit'])
        # plt.show()

        q = 0
        if 'orig' in fitnesses:
            del fitnesses['orig']
        for key in fitnesses.keys():
            # if q <=0:
            #     q+=1
            #     continue
            print(key)
            if model_name is None:
                collection_name = 'mods2/mutateDistance/' + repr + "_" + str(key) + "_collection"
            else:
                collection_name = os.path.join(path_base1, model_name, '_collection')
            corrName = os.path.join(path_base1, model_name, '_corrs')

            if os.path.exists(collection_name) and model_name is None:
                with open(collection_name, 'rb') as fi:
                    collection = pickle.load(fi)
            else:
                fit = fitnesses[key]
                lat = list(latents[key])
                lat_enc = list(latents_encoder[key])
                muts = mutants[key]
                muts_length = len(muts)
                print("mutants: ", muts_length)
                muts_dec = mutants_decoder[key]  ## dec
                fit_dec = fitnesses2[key]
                sort = sorted(list(zip(fit, lat, lat_enc, muts, muts_dec, fit_dec)), key=lambda x: x[0])  ## dec
                sort = [tupl for tupl in sort if len(tupl[3]) <= maxlen]
                fit, lat, lat_enc, muts, muts_dec, fit_dec = list(zip(*sort))  ## dec
                # fit = list(fit)
                lat = np.array(lat)
                lat_enc = np.array(lat_enc)
                avg_dists_to_not_worse = []
                avg_dists_to_not_worse_enc = []
                lat_enc_dists = []
                muts_set = set()
                fitness_vector = []
                fitness_dec_vector = []
                dec_fitn_diff = []
                for i, score in enumerate(sort[:-1]):
                    if i % 100 == 0:
                        print(i, "/", muts_length, flush=True)
                    # print(i)

                    current_mut = score[3]
                    current_fit = score[0]  # round(score[0],4)
                    if current_mut in muts_set or current_fit <= -1.0:
                        # print(current_mut)
                        continue
                    muts_set.add(current_mut)

                    fitness_vector.append(current_fit)
                    fitness_dec_vector.append(score[5])
                    dec_fitn_diff.append(abs(score[5] - score[0]))
                    to_compare = lat[i + 1:]
                    avg_dist_to_not_worse = np.mean(np.sqrt(np.sum((to_compare - score[1]) ** 2, axis=-1)))
                    avg_dists_to_not_worse.append(avg_dist_to_not_worse)

                    to_compare_enc = lat_enc[i + 1:]
                    avg_dist_to_not_worse_enc = np.mean(np.sqrt(np.sum((to_compare_enc - score[2]) ** 2, axis=-1)))
                    avg_dists_to_not_worse_enc.append(avg_dist_to_not_worse_enc)
                    lat_enc_dists.append(np.sum((score[1] - score[2]) ** 2, axis=-1))
                collection = (
                fitness_vector, fitness_dec_vector, avg_dists_to_not_worse, avg_dists_to_not_worse_enc, lat_enc_dists,
                muts_set)
                with open(collection_name, 'wb') as f:
                    pickle.dump(collection, f)
            fitness_vector, fitness_dec_vector, avg_dists_to_not_worse, avg_dists_to_not_worse_enc, lat_enc_dists, muts_set = collection
            print(len(muts_set))
            sp_corr = spearmanr(fitness_vector, avg_dists_to_not_worse)[0]
            sp_corr_enc = spearmanr(fitness_vector, avg_dists_to_not_worse_enc)[0]
            name = namemap[key]
            corrs[name] = sp_corr
            corrs[name + 'sp'] = sp_corr_enc
            print(key)
            print("corr: ", sp_corr)
            print("corr (Encoder pass): ", sp_corr_enc)

            with open(corrName, 'wb') as f:
                pickle.dump([sp_corr, sp_corr_enc], f)

            # fig = plt.figure()
            # ax = fig.add_subplot(121)  # , projection='3d')
            # fig, axes = plt.subplots(nrows=1, ncols=2, sharey=True)  # , figsize=(5, 3))
            # fig.set_figheight(4.8)
            # fig.set_figwidth(8)
            #
            # axes[0].scatter(fitness_vector, avg_dists_to_not_worse, s=1)
            #
            # axes[0].set_title("Sampled latent points, %s %s" % (repr + name, model_name.split('_')[3]))
            # axes[0].set_ylabel('Avg. distance from not worse solutions', fontsize=12, labelpad=10)
            # axes[0].set_yscale('log')
            # axes[0].set_xlabel('Fitness', fontsize=12, labelpad=10)
            # # plt.show()
            #
            # # fig = plt.figure()
            # # ax = fig.add_subplot(122)  # , projection='3d')
            # axes[1].scatter(fitness_vector, avg_dists_to_not_worse_enc, s=1)
            # # name = namemap[key]
            # axes[1].set_title("Representation specific points, %s %s" % (repr + name, model_name.split('_')[3]))
            # # ax.set_title("Fitness distance correlation for specific points, " + name)
            # # axes[1].set_ylabel('avg. distance from not worse solutions')
            # axes[1].set_yscale('log')
            # axes[1].set_xlabel('Fitness', fontsize=12, labelpad=10)
            # plt.savefig(os.path.join(path_base1, model_name, '_FDC_plot'),
            #             dpi=300,
            #             transparent=False, bbox_inches='tight')
            # # plt.show()
            # plt.clf()

            # fig = plt.figure()
            # ax = fig.add_subplot(111)  # , projection='3d')
            # ax.scatter(fitness_vector, lat_enc_dists, s=1)
            # name = namemap[key]
            # ax.set_title("Distance between p and sp(p), %s " % repr + name)
            # plt.ylabel('distance')
            # plt.yscale('log')
            # plt.xlabel('fitness')
            # plt.show()

            # fig = plt.figure()
            # ax = fig.add_subplot(111)  # , projection='3d')
            # ax.scatter(fitness_vector, dec_fitn_diff, s=1)
            # name = namemap[key]
            # ax.set_title("Fitness difference after additional enc+dec pass, " + name)
            #
            # plt.ylabel('fitness diff')
            # plt.xlabel('fitness base')
            # plt.show()
            # plt.clf()
        # fig = plt.figure(figsize=(6.4, 4.8))
        # ax = fig.add_subplot(111)  # , projection='3d')
        # vals = np.arange(len(corrs))
        # corr_values_points = [cr + 0.05 for cr in corrs.values()]
        # points = list(zip(list(vals - 0.25), corr_values_points))
        # names = list(corrs.keys())
        # names = [re.sub(' +', ' ', nam.replace('model', '')) for nam in names]
        # for i in range(len(names)):
        #     plt.annotate(names[i], points[i])
        # ax.scatter(vals, list(corrs.values()), s=30)
        # # plt.title('Finess distance correlation for %s models' % repr, fontsize=12)
        # plt.ylabel("Fitness distance correlation", fontsize=12, labelpad=10)
        # plt.xlabel("%s models" % repr, fontsize=12, labelpad=10)
        # plt.tick_params(
        #     axis='x',  # changes apply to the x-axis
        #     which='both',  # both major and minor ticks are affected
        #     bottom=False,  # ticks along the bottom edge are off
        #     top=False,  # ticks along the top edge are off
        #     labelbottom=False)
        # plt.xlim(-1, vals[-1] + 1)
        # plt.ylim(-1, 1)
        # plt.savefig(
        #     os.path.join(path_base1, model_name, '_collection'),
        #     dpi=300,
        #     transparent=False, bbox_inches='tight')
        # # plt.show()
        # plt.clf()


    FDC(df, model_name=name)

# runFDCforModel(name='model_f1_None_64_twoLayer_Bidir_0-0n_16',  data_folder="mods2/")

if __name__ == '__main__':
    # from pathlib import Path
    #
    #
    #
    # home = str(Path.home())
    # datafolder = os.path.join(home, 'dataFolder2')
    # collectFDCCorrs(datafolder)
    drawFDCs()