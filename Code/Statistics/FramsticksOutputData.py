import pandas as pd
from datetime import datetime
import numpy as np
import re
import seaborn as sns
import matplotlib.pyplot as plt
import glob
import os
import pickle
from collections import Counter
date_format = '%Y%m%d%H%M%S'
directory_base = 'logbooks'
def select_lines(file_name):
    col_names = ['num_cr', 'num_gen', 'worst', 'avg', 'best']
    df = pd.DataFrame(columns=['start', 'end', 'time', 'param', 'run'] + col_names)
    with open(file_name, 'r') as f:
        since_start = -9999
        since_finish = -9999
        obj = {}
        obj_list = []
        logged_values = []
        parameterization = 0
        run = 0
        for line in f.readlines():

            if line.startswith('C:\\'):
                regex = '[a-z]*_[0-9]+_[0-9]+'
                match = re.search(regex, line)
                if match is not None:
                    _, parameterization, run = match.group(0).split('_')

            if 'start' in line.split(' ')[0]:
                obj = {} #initialize a fresh object
                since_start = 0
            elif 'finish' in line.split(' ')[0]:
                since_finish = 0
                # run += 1

            elif since_start is not None and since_start==3:
                date = ''.join([c for c in line if c != ' ' and c!= chr(0)]).split('.')[0]
                obj['start_time'] = datetime.strptime(date, date_format)
            elif since_finish is not None and since_finish==3:
                date = ''.join([c for c in line if c != ' ' and c!= chr(0)]).split('.')[0]
                obj['end_time'] = datetime.strptime(date, date_format)

                data = np.stack(logged_values)
                logged_values = []
                obj['time'] = obj['end_time'] - obj['start_time']

                info = [obj['start_time'], obj['end_time'], obj['time'], parameterization, run]
                data = np.apply_along_axis(lambda x: np.insert(x, 0, info), 1, data)
                new_part_df = pd.DataFrame(data, columns=df.columns)
                df = df.append(new_part_df, ignore_index=True)

                obj_list.append(obj) #save the object

            elif '[LOG]' in line:
               logged_values.append(line.split()[2:])

            since_start += 1
            since_finish += 1

        return df

def fix_dtypes(df):
    numerics = [ 'run', 'num_cr', 'worst', 'avg', 'best']
    # dates = ['start', 'end']

    for col in numerics:
        df[col] = pd.to_numeric(df[col])
    # for col in dates:
    #     df[col] = pd.to_datetime(df[col], format='%Y-%m-%d %H:%M:%S')
    # df['time'] = pd.to_timedelta(df['time'])
    return df

def plot_avg_fitness(df):
    sns.set(style='darkgrid')
    plot = sns.violinplot(x=df['param'], y=df['avg'])
    plot.set_title("Average scores for all parameterizations")
    plot.set_ylabel("Average score")
    plot.set_xlabel("Parameterizations")
    plt.show()


def parmName_f(row):
    # nameMap = {'f9':'f9', 'f1':'f1', '3': 'phenotype', '0n': 'None', '3f': 'fitness', '3L' : 'levenshtein', '3l':'levenshtein'}

    parmName = 'invalid'
    nameMap = {'f9':'f9', 'f1':'f1', '3': 'phenotype', '0n': 'standard', '0.0n': 'standard', '3f': 'fitness', '3L' : 'levenshtein', '3l':'levenshtein'}
    parm = row.param
    suffix = ''
    if 'untrained' in parm:
        suffix = "untrained"

    for k in nameMap.keys():
        if k in parm:
            parmName = nameMap[k]
            if 'cmaes' in parm:
                parmName = parmName+'_cmaes'
            # if parmName == 'f9':
            #     if parm.endswith('True'):
            #         parmName = 'f9_pythonMut'
            #     else:
            #         parmName = 'f9_origMut'
            # if parmName == 'f1':
                # if parm.endswith('True'):
                #     parmName = 'f1_pythonMut'
                # else:
                #     parmName = 'f1_origMut'

    # if not parmName.startswith('f'):
    #     parmName = parm.split("_")[-3]

    if suffix == 'untrained':
        return suffix
    return parmName+suffix

def clr_f(row):
    cMap = {'f9_pythonMut': 'purple', 'f1_pythonMut':'purple', 'phenotype': 'g', 'fitness':'r', 'levenshtein':'orange', 'None':'b', 'f9_origMut': 'k', 'f1_origMut' :'k'}
    parmName = row.parmName
    return cMap[parmName]




import matplotlib as mpl

plt.style.use(['science', 'no-latex', 'grid'])


def plot_f(ax, xlabel, ylabel, fontsize, name):
    # ax.xaxis.set_tick_params(which='major', size=10, width=1, direction='in', top='on')
    # ax.xaxis.set_tick_params(which='minor', size=6, width=1, direction='in', top='on')
    # ax.yaxis.set_tick_params(which='major', size=10, width=1, direction='in', right='on')
    # ax.yaxis.set_tick_params(which='minor', size=6, width=1, direction='in', right='on')
    ax.set_xlabel(xlabel, fontsize=fontsize, labelpad=10)
    ax.set_ylabel(ylabel, fontsize=fontsize, labelpad=10)

    ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(30))
    ax.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(10))
    ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(0.2))
    ax.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(0.1))
    ax.legend(frameon=False, fontsize=fontsize)
    plt.savefig('C:\\Users\\Piotr\\Desktop\\Magisterka\\praca mag\\obrazki praca\\auto_generated\\%s.png'%name, dpi=300,
                transparent=False, bbox_inches='tight')
    plt.show()
    plt.clf()


def plot_gens(df):
    # style_map = {'t3': '-', 't5':'--'}
    df['parmName'] = df.apply(parmName_f, axis=1)
    # df['clr'] = df.apply(clr_f, axis=1)

    df['model name'] = df['parmName']
    df['tournament size'] = df['tsize']
    df['SP operators'] = df['useEncoder']
    print(df.groupby(['model name', 'tsize', 'mut_magnitude', 'useEncoder','cmaes', 'cells']).size())
    Means = df.groupby(['model name', 'tsize', 'mut_magnitude', 'useEncoder', 'cmaes', 'cells'])['best'].mean()  # [df['model name'].unique()]
    medians = df.groupby(['model name', 'tsize', 'mut_magnitude', 'useEncoder', 'cmaes', 'cells'])['best'].median()  # [df['model name'].unique()]
    maxes = df.groupby(['model name', 'tsize', 'mut_magnitude', 'useEncoder', 'cmaes', 'cells'])['best'].max()  # [df['model name'].unique()]
    print(Means)
    print(medians)
    print(maxes)
    fitn1 = df[(df['model name']=='fitness') & (df['useEncoder']=='True')]
    fitn2 = df[(df['model name']=='fitness') & (df['useEncoder']=='False')]
    pheno1 = df[(df['model name']=='phenotype') & (df['useEncoder']=='True')]
    standard1 = df[(df['model name'] == 'standard') & (df['useEncoder'] == 'True')]
    standard2 = df[(df['model name'] == 'standard') & (df['useEncoder'] == 'False')]
    f9 = df[df['model name'] == 'f9']
    from scipy import stats
    # sns.lineplot(data=df, x='tsize', y='best', hue='parmName', ci=None,
    #              palette={'f9_pythonMut': 'purple', 'phenotype': 'g', 'fitness': 'r', 'levenshtein': 'orange',
    #                       'None': 'b', 'f9_origMut': 'k'},
    #              legend='brief')
    fig = plt.figure(figsize=(6.4, 4.8))
    ax= sns.lineplot(data=df, x='gen', y='best', hue='model name', estimator='mean', style='cells', ci=95,
                 # palette=sns.color_palette(n_colors=3),
                 palette= {'f1':'k', 'f9': 'k', 'standard':'#1f77b4', 'untrained':'purple', 'fitness':'r', 'phenotype':'g',
                           'phenotype_cmaes': 'g', 'fitness_cmaes':'r', 'standard_cmaes':'#1f77b4'},
                 # palette={'f9_pythonMut': 'purple', 'f1_pythonMut':'purple', 'phenotype': 'g', 'fitness':'r', 'levenshtein':'orange', 'None':'b', 'f9_origMut': 'k', 'f1_origMut' :'k'},
                 legend='brief')
    plot_f(ax, "Generations", "Mean best", 12, "f9 new res")
    # df['model name'] = df['model name'] + df['tsize'].map(str)
    # Means = df.groupby('model name')['best'].mean()[df['model name'].unique()]
    # medians = df.groupby('model name')['best'].median()[df['model name'].unique()]
    # maxes = df.groupby('model name')['best'].max()[df['model name'].unique()]
    # print(Means)
    # print(medians)
    # print(maxes)
    # sns.violinplot(data=df, x='model name', y='best')
    # plt.scatter(x=np.array(range(len(Means))), y=Means, c="k")
    # plt.scatter(x=np.array(range(len(maxes))), y=maxes, c="k", marker='X')
    # for parm in df['param'].unique():
    #     gptemp = df[df['param'] == parm]
    #     # gp = gptemp.groupby(['gen']).mean()
    #     parmName = 'invalid'
    #
    #
    #     # sns.scatterplot(data=gptemp, x='gen', y='best', label=parmName, color=cMap[parmName], s=15)
    #     # plt.plot(gp['num_cr'], gp['best'], label=parm)
    # # plt.legend(loc='upper left')
    # # plt.title("true fitness, do not use encoder, different tourn sizes")
    # plt.title("evolution fitness, mean best vs gen")
    # plt.legend(fontsize=20)
    # plt.


def plot_best_fitness(df):
    sns.set(style='darkgrid')
    plot = sns.violinplot(x=df['param'], y=df['best'])
    plot.set_title('Best scores for all parametrizations')
    plot.set_ylabel("Best score")
    plot.set_xlabel("Parameterizations")
    plt.show()

def plot_computation_time_for_parameterizations(df):
    sns.set(style='darkgrid')
    plot = sns.violinplot(x=df['param'], y=df['time'] / np.timedelta64(1, 's'))
    plot.set_title('Duration of experiments for all parametrizations')
    plot.set_ylabel("Duration of experiment")
    plot.set_xlabel("Parameterizations")
    plt.show()

def plot_multiline_fitness(df):
    sns.set(style='darkgrid')
    # tmp_df = df[df['param']==1]
    #for i, c in zip(df.param.unique(), ['Reds_r', 'Blues_r', 'Greens_r', 'Reds_r']):
    for i in df.param.unique():

        tmp_df = df[df['param']==i]

        palette = dict(zip(tmp_df.run.unique(),
                           sns.color_palette('Blues_r', 15)))
        # plt.figure(figsize=(15, 8))
        plot = sns.relplot(x='num_cr', y='best', palette=palette, #hue='run',
                kind='line', data=tmp_df)
        plot.fig.set_size_inches(14, 10)
        plot.set_axis_labels("Number of creatures tested", "Best score in current population")

        plt.title("Number of creatures and their score for %s evolution" %i)

        plt.savefig(os.path.join("plots/evol_plots",i+'_evol_plot.png'))
        plt.clf()
    # plt.show()

def get_latent_df_scores():
    load = True
    num = -1
    # col_names = ['num_cr', 'num_gen', 'worst', 'avg', 'best']
    df = pd.DataFrame()
    cnt = Counter()
    for fname in glob.glob(f'{directory_base}/**/logbook_*', recursive=True):
        repr = 'f9'
        if repr not in fname:
            continue

        dir, f = os.path.split(fname)
        dir, name = os.path.split(dir)
        num += 1
        print(fname)
        with open(fname, 'rb') as ff:
            x  = pickle.load(ff)
            config = None
            tsize = 3
            cmaes= ''
            config = {}
            fitness_len_weight, fitness_len_max_value, fitness_max_len, fitness_len_chars_sub = [0.0, 0.0, -1, '']
            fitness_min_value = -1
            # print(len(x))
            # continue
            if len(x) == 0:
                continue
            if len(x) ==6:
                logbook, name, fake_f, use_enc, keepbest, fake_m = x

            elif len(x) == 7:
                logbook, name, fake_f, use_enc, keepbest, fake_m, config = x
                tsize = config['tourney_size']
            elif len(x) == 10:
                logbook, name, fake_f, use_enc, keepbest, fake_m, config, fitness_len_weight, fitness_len_max_value, fitness_max_len = x
                tsize = config['tourney_size']
            elif len(x) == 11:
                logbook, name, fake_f, use_enc, keepbest, fake_m, config, fitness_len_weight, fitness_len_max_value, fitness_max_len, fitness_len_chars_sub = x
                tsize = config['tourney_size']
            elif len(x) == 12:
                logbook, name, fake_f, use_enc, keepbest, fake_m, config, fitness_len_weight, fitness_len_max_value, fitness_max_len, fitness_len_chars_sub, fitness_min_value = x
                tsize = config['tourney_size']
            elif len(x) == 5:
                logbook, name, fake_f, use_enc, keepbest = x
                fake_m = False
            else:
                logbook = x
                name = fname
                fake_m = False
                fake_f = False
                use_enc = False
                keepbest = False
        try:

            nam_base = '_'.join(name.split("_"))
            nam_split = nam_base.split("_")
            suffix = ''

            if not nam_split[0].startswith('f') and config['test'] == 99:
                suffix = 'untrained'
            if fitness_min_value != -1:
                continue

            # print(nam_split)
            # if config is not None or (config is not None and config['tourney_size'] != 5):
            #     continue
            # if nam_split[0].startswith('f9') and not fake_m: # f9 mutation
            #     continue
            # if (not nam_split[0].startswith('f9')) and config['mut_magnitude'] != 0.2:
            #     continue
            # if (not nam_split[0].startswith('f9')) and not (tsize == 1 or suffix == 'untrained'):
            #     continue
            if tsize == 1 or suffix == 'untrained':
                continue
            # if 'cmaes' in config and config['cmaes']:
            #     continue
            # if  not use_enc and (not nam_split[0].startswith('f')):
            #     continue
            #CMAES FILT
            # if 'cmaes' in config and config['cmaes']:
            #
            #     cmaes = 'cmaes'
            # elif (not nam_split[0].startswith('f')) or (tsize==1):
            #     continue



            # print(config['locality_power'])
            # if not fitness_len_weight == 0.01:
            #     continue
            # if not fitness_max_len == 20:
            #     continue
            # if not fitness_len_max_value == 0.2:
            #     continue
            # if fake_f: # FAKE F filter
            #     continue
            # if nam_split[0].startswith('f'):
            #     continue
            # if ((config['test'] >= 5 and config['test'] <= 9)): #or (config['test']==0 and config['locality_type']=='noloc')) :
            #     continue
            # if (use_enc) and (not nam_split[0].startswith('f')):
            #     continue
            # if 'cmaes' not in config and (not nam_split[0].startswith('f')):
                # if nam_split[0].startswith('f'):
                #     x=0
                # continue
            # if 'cmaes' not in config and tsize != 3:
            #     continue
            df_temp = pd.DataFrame(logbook).rename(columns={'max': 'best', 'min':'worst', 'nevals':'num_cr'})
            # print(df_temp)
            # if df_temp['best'].max() < 0.0:
            #     continue
            # filtering
            # name = model_name
            df_temp['num_cr'] = df_temp['gen'] * 50 # POPULSIZE INSTEAD OF 50 should be there
            # df_temp = df_temp[df_temp['gen']<=100]
            df_temp['tsize'] = tsize
            if nam_split[0].startswith('f'):
                use_enc = False
            df_temp['useEncoder'] = str(use_enc)
            print(name, df_temp['best'].max())

            # if 'nolatent' in nam_base:
            #     x=0
            # CMAES CUM MAX
            # df_temp = df_temp.sort_values(by='gen', ascending=True)
            # df_temp['best'] = df_temp['best'].cummax()

            # df_temp = df_temp[df_temp['best'] == df_temp['best'].max()].iloc[0:1] # THIS LINE ENABLE/DISABLE
            # df_temp = df_temp[df_temp['gen'] == df_temp['gen'].max()]

            if len(nam_split) > 1 and nam_split[6] == "0":
                df_temp['param'] = 'lat' +'_'+str(cmaes) +  '_'+str(fake_f)
                # df_temp = df_temp[df_temp['gen'] == df_temp['gen'].max()]
            elif nam_split[0].startswith('f'):
                df_temp['param'] = nam_split[0] + '_'+str(fake_f)+str(fake_m)
                # df_temp = df_temp[df_temp['gen'] == df_temp['gen'].max()]
            # elif nam_split[3] == "True":
            #     df_temp['param'] = 'f1_rand'
            #     df_temp = df_temp[df_temp['best'] == df_temp['best'].max()]
            # elif nam_split[6] == "True":
            #     df_temp['param'] = 'lat_rand'
            #     df_temp = df_temp[df_temp['best'] == df_temp['best'].max()]
            else:
                nam_base=nam_base.replace("-", ".")
                nam_base=nam_base.replace("l", "L")
                nam_base=nam_base.replace("00", "0")
                # df_temp['param'] = nam_base + '_' +str(cmaes) + '_'+str(fake_f)
                # if 'n' not in nam_split[6]:
                #     continue
                # else:
                #     if tsize != 1 and suffix != 'untrained':
                #         continue
                df_temp['param'] = nam_base.split("_")[6] + '_' +str(cmaes) + '_'+str(fake_f) + "_" + suffix

            # df_temp['param'] = nam_base
            # nam_base = 'f1' if 'nolatent' in name else 'latent'
            # nam_base += '_random' if 'True' in name else ''
            cnt[nam_base+"__"+suffix+"__" + str(tsize)+'_'+str(use_enc) + "__" + str(config['mut_magnitude']) +"__"+str(config['cmaes'])+"__" + str(fitness_min_value)] += 1
            # cnt[nam_base+str(config['mut_magnitude'])+str(tsize)] += 1
            print(len(df_temp))
            df_temp['run'] = 0
            df_temp['mut_magnitude'] = config['mut_magnitude']
            df_temp['cmaes'] = cmaes
            df_temp['cells'] = config['cells']
            df = pd.concat([df, df_temp], axis=0)
        except Exception as e:
            print('err', nam_split, tsize, e)
    print(cnt)
    # while load:
    #     num += 1
    #     try:
    #         df_temp = pd.read_csv('evolution_experiments_latent/logbook_'+str(num)).rename(columns={'max': 'best', 'min':'worst', 'nevals':'num_cr'})
    #         df_temp['num_cr'] = df_temp['gen'] * 50
    #         df_temp['run'] = 0
    #         df_temp['param']=1
    #         df  = pd.concat([df,df_temp], axis=0)
    #     except:
    #         load = False
    #         continue
    return df

df_latent = get_latent_df_scores()
# df = select_lines('outputFile.txt')
#
# df = pd.concat([df, df_latent], axis=0)
df = df_latent

df = fix_dtypes(df)

# df_cp = df[df['param']==0].copy()
# df_cp['param'] = 2
# df_cp = df_cp[df_cp['num_cr']<2500]
# df = pd.concat([df, df_cp], axis=0)

plot_gens(df)
df = df[df['gen'] == df['gen'].max()]

# plot_multiline_fitness(df)
plot_avg_fitness(df)
plot_best_fitness(df)
# plot_computation_time_for_parameterizations(df)