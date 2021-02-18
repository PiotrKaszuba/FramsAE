import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
import pickle
import seaborn as sns
from mpl_toolkits.mplot3d.axes3d import Axes3D
from sklearn import preprocessing
import matplotlib.transforms as transforms
from collections import defaultdict
from scipy.stats import spearmanr
import re





path_base1 = "mods2/mutateDistance/"
path_base2="**/*mutDist*"
path_base = path_base1+path_base2
repr = 'f1'
def gatherScores(name = None):
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
df = gatherScores()
from Code.drawDistributions import plot_f, c1
def FDC(df, model_name = None):

    latents = defaultdict(lambda: None)
    latents_encoder = defaultdict(lambda: None)
    fitnesses = defaultdict(list)
    fitnesses2 = defaultdict(list)
    mutants = defaultdict(list)
    mutants_decoder = defaultdict(list) # dec
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
        # if len(mutants[loc]) > 10000:
        #     continue
        fitnesses[loc].extend(row['fitnesses'])
        fitnesses2[loc].extend(row['fitnesses'])
        mutants[loc].extend(row['mutants'])

        mutants_decoder[loc].extend(row['mutants']) ### dec
        if loc != 'orig':
            if latents[loc] is None:
                latents[loc] = row['latents']
            else:
                latents[loc] = np.concatenate([latents[loc], row['latents']], axis=0)

            if latents_encoder[loc] is None:
                latents_encoder[loc] = row['latents_encoder']
            else:
                latents_encoder[loc] = np.concatenate([latents_encoder[loc], row['latents_encoder']], axis=0)
    namemap = {'noloc': 'standard model', 'levenshtein': 'levenshtein', 'dissim': 'phenotype model', 'fitness': 'fitness model'}
    corrs = {}
    corrs_enc = {}
    print(names)
    df2 = pd.DataFrame()
    for key in fitnesses.keys():
        fit = fitnesses[key]
        lf = len(fit)
        fit = [f for f in fit if f > -1]
        print(len(fit)/lf)
        print(key, len(fit), np.mean(fit), np.median(fit), np.max(fit))
        df_temp = pd.DataFrame(pd.Series(fit, name='fit'))

        df_temp['name'] = key
        df2 = pd.concat([df2, df_temp])
    # sns.violinplot(x=df2['name'], y=df2['fit'])
    plt.show()

    q=0
    del fitnesses['orig']
    for key in fitnesses.keys():
        # if q <=0:
        #     q+=1
        #     continue
        print(key)
        if model_name is None:
            collection_name = 'mods2/mutateDistance/'+repr+"_"+str(key)+"_collection"
        else:
            collection_name = os.path.join(path_base1, model_name, '_collection')
        if os.path.exists(collection_name):
            with open(collection_name, 'rb') as fi:
                collection = pickle.load(fi)
        else:
            fit = fitnesses[key]
            lat = list(latents[key])
            lat_enc = list(latents_encoder[key])
            muts = mutants[key]
            muts_length = len(muts)
            print("mutants: ", muts_length)
            muts_dec = mutants_decoder[key] ## dec
            fit_dec = fitnesses2[key]
            sort = sorted(list(zip(fit, lat, lat_enc, muts, muts_dec, fit_dec)), key=lambda x: x[0]) ## dec
            sort = [tupl for tupl in sort if len(tupl[3]) <=maxlen]
            fit, lat, lat_enc, muts, muts_dec, fit_dec = list(zip(*sort)) ## dec
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
                if i%100==0:
                    print(i, "/",muts_length)
                # print(i)

                current_mut = score[3]
                current_fit = score[0]  # round(score[0],4)
                if current_mut in muts_set or current_fit <= -1.0:
                    # print(current_mut)
                    continue
                muts_set.add(current_mut)

                fitness_vector.append(current_fit)
                fitness_dec_vector.append(score[5])
                dec_fitn_diff.append(abs(score[5]-score[0]))
                to_compare = lat[i+1:]
                avg_dist_to_not_worse = np.mean(np.sqrt(np.sum((to_compare - score[1]) ** 2, axis=-1)))
                avg_dists_to_not_worse.append(avg_dist_to_not_worse)

                to_compare_enc = lat_enc[i + 1:]
                avg_dist_to_not_worse_enc = np.mean(np.sqrt(np.sum((to_compare_enc - score[2]) ** 2, axis=-1)))
                avg_dists_to_not_worse_enc.append(avg_dist_to_not_worse_enc)
                lat_enc_dists.append(np.sum((score[1] - score[2]) ** 2, axis=-1))
            collection = (fitness_vector, fitness_dec_vector, avg_dists_to_not_worse, avg_dists_to_not_worse_enc, lat_enc_dists, muts_set)
            with open(collection_name, 'wb') as f:
                pickle.dump(collection, f)
        fitness_vector, fitness_dec_vector, avg_dists_to_not_worse, avg_dists_to_not_worse_enc, lat_enc_dists, muts_set = collection
        print(len(muts_set))
        sp_corr = spearmanr(fitness_vector, avg_dists_to_not_worse)[0]
        sp_corr_enc = spearmanr(fitness_vector, avg_dists_to_not_worse_enc)[0]
        name = namemap[key]
        corrs[name] = sp_corr
        corrs[name+'sp'] = sp_corr_enc
        print(key)
        print("corr: ", sp_corr)
        print("corr (Encoder pass): ", sp_corr_enc)
        # fig = plt.figure()
        # ax = fig.add_subplot(121)  # , projection='3d')
        fig, axes = plt.subplots(nrows=1, ncols=2, sharey=True)#, figsize=(5, 3))
        fig.set_figheight(4.8)
        fig.set_figwidth(8)


        axes[0].scatter(fitness_vector, avg_dists_to_not_worse, s=1)

        axes[0].set_title("Sampled latent points, %s " %repr + name)
        axes[0].set_ylabel('Avg. distance from not worse solutions', fontsize=12, labelpad=10)
        axes[0].set_yscale('log')
        axes[0].set_xlabel('Fitness', fontsize=12, labelpad=10)
        # plt.show()

        # fig = plt.figure()
        # ax = fig.add_subplot(122)  # , projection='3d')
        axes[1].scatter(fitness_vector, avg_dists_to_not_worse_enc, s=1)
        # name = namemap[key]
        axes[1].set_title("Representation specific points, %s " %repr + name)
        # ax.set_title("Fitness distance correlation for specific points, " + name)
        # axes[1].set_ylabel('avg. distance from not worse solutions')
        axes[1].set_yscale('log')
        axes[1].set_xlabel('Fitness', fontsize=12, labelpad=10)
        plt.savefig('C:\\Users\\Piotr\\Desktop\\Magisterka\\praca mag\\obrazki praca\\auto_generated\\fdc_%s.png' % (repr + name),
                    dpi=300,
                    transparent=False, bbox_inches='tight')
        plt.show()
        plt.clf()

        fig = plt.figure()
        ax = fig.add_subplot(111)  # , projection='3d')
        ax.scatter(fitness_vector, lat_enc_dists, s=1)
        name = namemap[key]
        ax.set_title("Distance between p and sp(p), %s " %repr + name)
        plt.ylabel('distance')
        plt.yscale('log')
        plt.xlabel('fitness')
        plt.show()

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
    fig = plt.figure(figsize=(6.4,4.8))
    ax = fig.add_subplot(111)  # , projection='3d')
    vals = np.arange(len(corrs))
    corr_values_points = [cr+0.05 for cr in corrs.values()]
    points = list(zip(list(vals-0.25), corr_values_points ))
    names = list(corrs.keys())
    names = [re.sub(' +', ' ', nam.replace('model', '')) for nam in names]
    for i in range(len(names)):
        plt.annotate(names[i], points[i])
    ax.scatter(vals, list(corrs.values()), s=30)
    # plt.title('Finess distance correlation for %s models' % repr, fontsize=12)
    plt.ylabel("Fitness distance correlation", fontsize=12, labelpad=10)
    plt.xlabel("%s models" % repr, fontsize=12, labelpad=10)
    plt.tick_params(
        axis='x',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
        labelbottom=False)
    plt.xlim(-1, vals[-1]+1)
    plt.ylim(-1, 1)
    plt.savefig(
        'C:\\Users\\Piotr\\Desktop\\Magisterka\\praca mag\\obrazki praca\\auto_generated\\fdcAll_%s.png' % repr,
        dpi=300,
        transparent=False, bbox_inches='tight')
    plt.show()
    plt.clf()


FDC(df)
# df = df[(df['locality_power'] >= 1) | (df['locality_power']==0) ]
# df = df[(df['locality'] == 'levenshtein') | (df['learning'] == 'standard')]
# df = df[df['learning'] == 'loc_like_batch']
# df['power'] = df['power'].fillna(value=0.1)
df['locality'] = df['locality'].fillna(value='orig')
# df = df[((df['power'] >= 0.0) & (df['power'] <= 0.8)) | (df['power']==-1)]
print(df.columns)

measure = 'f'
aggregation = 'mean'

measure_dict = {'l': 'edit distance', 'f': 'fitness difference', 'd':'phenotype dissimilarity'}
aggregation_dict = {'median': 'Median', 'mean':'Mean'}
name_to_plot = 'avg_%s_%s' % (measure, aggregation)
for col in df.columns:
    if 'corr' in col:
        df[col] = list(map(lambda x: (np.mean(x)*4-2)/2, df[col]))


fig = plt.figure(figsize=(6.4,4.8))
ax = fig.add_subplot(111)#, projection='3d')
#
df['valid_mutants'] = df['valid_mutants']/(df['valid_mutants']+df['invalid_mutants'])
f5Df = df[df['locality'] == 'orig']

le2 = preprocessing.LabelEncoder()
f5Df['power_enc'] = le2.fit_transform(f5Df['power'])
f5Df['power_enc'] = f5Df.power_enc+1
# f5Df['power_enc'] = f5Df['power']


le = preprocessing.LabelEncoder()
df['power_enc'] = le.fit_transform(df['power'])
df['power_enc'] = df.power_enc+1
offset = lambda p: transforms.ScaledTranslation(p/72.,0, plt.gcf().dpi_scale_trans)
trans = plt.gca().transData
f1Df = df[df['locality'] == 'noloc']
f4Df = df[df['locality'] == 'levenshtein']
f9Df = df[df['locality'] == 'dissim']
f0Df = df[df['locality'] == 'fitness']
print(f1Df['valid_centroids'].mean())
print(f9Df['valid_centroids'].mean())
print(f0Df['valid_centroids'].mean())


f1=ax.scatter(f1Df['power_enc'], f1Df[name_to_plot], s=15, c=c1)#, transform=trans+offset(-24), s=15)
f4=ax.scatter(f4Df['power_enc'], f4Df[name_to_plot], s=15)#, transform=trans+offset(-8), s=15)
f9=ax.scatter(f9Df['power_enc'], f9Df[name_to_plot], s=15, c='g')#, transform=trans+offset(+8), s=15)
f0=ax.scatter(f0Df['power_enc'], f0Df[name_to_plot], s=15, c='r')#, transform=trans+offset(+24), s=15)
f5=ax.scatter(f5Df['power_enc'], f5Df[name_to_plot], s=15, c='k')#, transform=trans+offset(0), s=15)
grp1=f1Df.groupby(by=['power_enc']).mean()
grp2=f4Df.groupby(by=['power_enc']).mean()
grp3=f9Df.groupby(by=['power_enc']).mean()
grp4=f0Df.groupby(by=['power_enc']).mean()
grp5=f5Df.groupby(by=['power_enc']).mean()

ax.scatter(grp1.index, grp1[name_to_plot], c='k', marker='v')# transform=trans+offset(-24))
ax.scatter(grp2.index, grp2[name_to_plot], c='k', marker='v')# transform=trans+offset(-8))
ax.scatter(grp3.index, grp3[name_to_plot], c='k', marker='v')# transform=trans+offset(+8))
ax.scatter(grp4.index, grp4[name_to_plot], c='k', marker='v')# transform=trans+offset(+24))
ax.scatter(grp5.index, grp5[name_to_plot], c='k', marker='v')# transform=trans+offset(0))
ax.plot(grp1.index, grp1[name_to_plot], c=c1, marker='v')# transform=trans+offset(-24))
ax.plot(grp2.index, grp2[name_to_plot], c='orange', marker='v')# transform=trans+offset(-8))
ax.plot(grp3.index, grp3[name_to_plot], c='g', marker='v')# transform=trans+offset(+8))
ax.plot(grp4.index, grp4[name_to_plot], c='r', marker='v')# transform=trans+offset(+24))
ax.plot(grp5.index, grp5[name_to_plot], c='k', marker='v')# transform=trans+offset(0))
# grps1 = [f1Df[f1Df['power_enc']==val][name_to_plot] for val in grp1.index.unique()]
# grps2 = [f4Df[f4Df['power_enc']==val][name_to_plot] for val in grp1.index.unique()]
# grps3 = [f9Df[f9Df['power_enc']==val][name_to_plot] for val in grp1.index.unique()]
# grps4 = [f0Df[f0Df['power_enc']==val][name_to_plot] for val in grp1.index.unique()]
# grps5 = [f5Df[f5Df['power_enc']==val][name_to_plot] for val in list(grp1.index.unique())+[-1]]
# ax.boxplot(grps1, positions=np.arange(len(grps1))-offset(13)._t[0]+1, showmeans=True, meanline=True, widths=0.1)
#
# ax.boxplot(grps3, positions=np.arange(len(grps3))+offset(5)._t[0]+1, showmeans=True, meanline=True, widths=0.1)
# ax.boxplot(grps2, positions=np.arange(len(grps2))-offset(5)._t[0]+1, showmeans=True, meanline=True, widths=0.1)
# ax.boxplot(grps4, positions=np.arange(len(grps4))+offset(13)._t[0]+1, showmeans=True, meanline=True, widths=0.1)
# ax.boxplot(grps5, positions=np.arange(len(grps5))+offset(0)._t[0], showmeans=True, meanline=True, widths=0.1)
# plt.ylabel("Median local edit distance", fontsize=12)
# plt.xlabel('Mutation magnitude latent/base encoding', fontsize=12)
# plt.yscale('log')
l = [0.0]+list(le.classes_[0:14][:7])
l2=[0, 1,2,3,4,6,8,10]
l = [str(p)+"/"+str(i) for p,i in zip(l, l2)]
plt.xticks(np.arange(8), l)

# plt.title("f1, Valid mutants vs mutation magnitude")
# plt.xlabel('Mutation power')

plt.legend((f1,f9, f0, f5),
           ('standard', 'phenotype', 'fitness', '%s' % repr),
           scatterpoints=1,
           loc='upper left',
           ncol=2,
           fontsize=12, frameon=False)
plot_f(ax, "Mutation magnitude latent/base encoding", "%s local %s" % (aggregation_dict[aggregation], measure_dict[measure]), 12, "%s_%s_%s" %(repr, aggregation, measure), legend=False, draw_minor=False)
# print(le.classes_)
# ax.scatter( df['locality_enc'],
#              df['power'],
#              df['avg'],
#             alpha=0.4)
print(df)
print(df.columns)


df.loc[df['locality']=='noloc', 'locality'] = 'standard'
df.loc[df['locality']=='dissim', 'locality']= 'phenotype'
df = df[df['locality'].isin(['standard', 'phenotype', 'fitness'])]
ax=sns.violinplot(df['locality'], df['valid_centroids'], palette=['#1f77b4','r','g','m'])
# plt.xlabel("Model type", fontsize=12)
# plt.ylabel("Validity", fontsize=12)
# plt.title("f1, Validity of sampled genotypes vs models")
# plt.show()
plot_f(ax, "Model type", "Validity", 12, "centroid validity", False)