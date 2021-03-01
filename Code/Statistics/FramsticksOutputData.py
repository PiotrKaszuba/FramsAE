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
directory_base = 'logbooks_MASTERS'
repr = 'f9'
only_best = False # True - table / boxplot of final/best results / False -> per generation plot
use_operators = ['slp'] # filters out not present 'ordinary', 'slp'
use_native = False # filters out native
use_untrained = False # filters out untrained
untrained_test_numbers = [99] # those identify versions of autoencoders that were untrained
included_t_sizes = [3]
included_POPSIZES = [50]
included_cells = [64]
included_mut_magnitude = [0.2]
required_generations = 150
max_generations = 150

style_var = None#'tournament size' # separate lines + legend on plot -> otherwise values of this are mixed (cells, tsize, None)
plot_name = 'slp_operators_MASTERS' # for filename with '%repr %plot_name'

# filter out if True
custom_filter = lambda x : False#lambda df_temp: 'untrained' not in df_temp['param'].iloc[0] and df_temp['tsize'].iloc[0] == 3 #lambda x: False

def fix_dtypes(df):
    numerics = [ 'run', 'num_cr', 'worst', 'avg', 'best']

    for col in numerics:
        df[col] = pd.to_numeric(df[col])

    return df



def parmName_f(row):
    # nameMap = {'f9':'f9', 'f1':'f1', '3': 'phenotype', '0n': 'None', '3f': 'fitness', '3L' : 'levenshtein', '3l':'levenshtein'}

    parmName = 'invalid'
    nameMap = {'f9':'native f9', 'f1':'native f1', '3': 'reconstr+pheno', '0n': 'reconstr', '0.0n': 'reconstr', '3f': 'reconstr+fitness', '3L' : 'reconstr+levenshtein', '3l':'reconstr+levenshtein'}
    parm = row.param
    suffix = ''
    if 'untrained' in parm:
        suffix = "untrained"

    for k in nameMap.keys():
        if k in parm:
            parmName = nameMap[k]
            if 'cmaes' in parm:
                parmName = parmName+' cmaes'
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

# def clr_f(row):
#     cMap = {'f9_pythonMut': 'purple', 'f1_pythonMut':'purple', 'phenotype': 'g', 'fitness':'r', 'levenshtein':'orange', 'None':'b', 'f9_origMut': 'k', 'f1_origMut' :'k'}
#     parmName = row.parmName
#     return cMap[parmName]


import matplotlib as mpl

plt.style.use(['science', 'no-latex', 'grid'])
from matplotlib.patches import PathPatch
def adjust_box_widths(g, fac):
    """
    Adjust the widths of a seaborn-generated boxplot.
    """

    # iterating through Axes instances
    for ax in g.axes:

        # iterating through axes artists:
        for c in ax.get_children():

            # searching for PathPatches
            if isinstance(c, PathPatch):
                # getting current width of box:
                p = c.get_path()
                verts = p.vertices
                verts_sub = verts[:-1]
                xmin = np.min(verts_sub[:, 0])
                xmax = np.max(verts_sub[:, 0])
                xmid = 0.5*(xmin+xmax)
                xhalf = 0.5*(xmax - xmin)

                # setting new width of box
                xmin_new = xmid-fac*xhalf
                xmax_new = xmid+fac*xhalf
                verts_sub[verts_sub[:, 0] == xmin, 0] = xmin_new
                verts_sub[verts_sub[:, 0] == xmax, 0] = xmax_new

                # setting new width of median line
                for l in ax.lines:
                    if np.all(l.get_xdata() == [xmin, xmax]):
                        l.set_xdata([xmin_new, xmax_new])

def plot_f(ax, xlabel, ylabel, fontsize, name, useTicksX = True, useTicksY=True, legend_loc=None, labelpad=10):

    # ax.xaxis.set_tick_params(which='major', size=10, width=1, direction='in', top='on')
    # ax.xaxis.set_tick_params(which='minor', size=6, width=1, direction='in', top='on')
    # ax.yaxis.set_tick_params(which='major', size=10, width=1, direction='in', right='on')
    # ax.yaxis.set_tick_params(which='minor', size=6, width=1, direction='in', right='on')
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=fontsize, labelpad=labelpad)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=fontsize, labelpad=labelpad)




    if useTicksX:
        ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(30))
        ax.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(10))
    if useTicksY:
        ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(0.2))
        ax.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(0.1))
    if legend_loc:
        ax.legend(frameon=False, fontsize=fontsize, loc=legend_loc)
    else:
        ax.legend(frameon=False, fontsize=fontsize)
    plt.savefig('C:\\Users\\Piotr\\Desktop\\Magisterka\\praca mag\\obrazki praca\\auto_generated\\article\\%s.pdf'%name, dpi=300,
                transparent=False, bbox_inches='tight')
    plt.show()
    plt.clf()


def plot_gens(df):
    # style_map = {'t3': '-', 't5':'--'}
    df['parmName'] = df.apply(parmName_f, axis=1)
    # df['clr'] = df.apply(clr_f, axis=1)

    df['model name'] = df['parmName']
    df['tournament size'] = df['tsize']
    op_map = {'False':'ordinary', 'True':'SLP'}
    df['operators'] = df['useEncoder'].apply(lambda x: op_map[x])
    df['SP operators'] = df['useEncoder']
    if only_best:
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

    pallette = {'native f1':'k', 'native f9': 'k', 'reconstr':'#1f77b4', 'untrained':'purple', 'reconstr+fitness':'r', 'reconstr+pheno':'g',
                               'reconstr+pheno cmaes': 'g', 'reconstr+fitness cmaes':'r', 'reconstr cmaes':'#1f77b4'}
    if not only_best:
        fig = plt.figure(figsize=(4, 3.2))
        ax= sns.lineplot(data=df, x='gen', y='best', hue='model name', estimator='mean', style=style_var, ci=95,
                     # palette=sns.color_palette(n_colors=3),
                     palette= pallette,
                     # palette={'f9_pythonMut': 'purple', 'f1_pythonMut':'purple', 'phenotype': 'g', 'fitness':'r', 'levenshtein':'orange', 'None':'b', 'f9_origMut': 'k', 'f1_origMut' :'k'},
                     legend='brief')
        plot_f(ax, "Generations", "Mean best fitness", 12, "%s_%s" % (repr, plot_name), labelpad=6)
    else:
        fig = plt.figure(figsize=(4, 4))
        ax = sns.boxplot(data=df, x='operators', hue='model name', y='best', meanline=True, showfliers=False, order=['ordinary', 'SLP'],
                    palette=pallette)

        ax.set_ylim([0.0, 1.35])
        adjust_box_widths(fig, 0.9)
        if repr == 'f1':  # beda obok siebie wiec dla f9 nie marnujemy miejsca na skale pionowa
            plot_f(ax, '%s operators' % repr, 'Best fitness', 12, "%s_%s" % (repr, plot_name), useTicksX=False,
                   legend_loc='lower center', labelpad=8)
        else:
            ax.set_yticklabels([])
            # ax.get_yaxis().set_visible(False)
            ax.set_ylabel('', fontsize=12, labelpad=10)
            plot_f(ax, '%s operators' % repr, None, 12, "%s_%s" % (repr, plot_name), useTicksX=False,
                   legend_loc='lower center', labelpad=8)


        # ax = sns.boxplot(data=df, x='model name', y='best')
        # plot_f(ax, 'Model', 'Mean best', 12, "%s %s" % (repr, plot_name) )
    # df['model name'] = df['model name']
    # Means = df.groupby('model name')['best'].mean()[df['model name'].unique()]
    # medians = df.groupby('model name')['best'].median()[df['model name'].unique()]
    # maxes = df.groupby('model name')['best'].max()[df['model name'].unique()]
    # print(Means)
    # print(medians)
    # print(maxes)
    # plt.scatter(x=np.array(range(len(Means))), y=Means, c="k")
    # plt.scatter(x=np.array(range(len(maxes))), y=maxes, c="k", marker='X')
    # for parm in df['param'].unique():
    #     gptemp = df[df['param'] == parm]
    #     # gp = gptemp.groupby(['gen']).mean()
    #     parmName = 'invalid'


        # sns.scatterplot(data=gptemp, x='gen', y='best', label=parmName, color=cMap[parmName], s=15)
        # plt.plot(gp['num_cr'], gp['best'], label=parm)
    # plt.legend(loc='upper left')
    # plt.title("true fitness, do not use encoder, different tourn sizes")
    # plt.title("evolution fitness, mean best vs gen")
    # plt.legend(fontsize=20)
    # plt.show()


def get_latent_df_scores():
    load = True
    num = -1
    # col_names = ['num_cr', 'num_gen', 'worst', 'avg', 'best']
    df = pd.DataFrame()
    cnt = Counter()
    for fname in glob.glob(f'{directory_base}/**/logbook_*', recursive=True):

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
            POPSIZE = 50
            cmaes= ''
            mut_magnitude = -1
            config = {}
            cells = 64
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
            elif len(x) == 10:
                logbook, name, fake_f, use_enc, keepbest, fake_m, config, fitness_len_weight, fitness_len_max_value, fitness_max_len = x
            elif len(x) == 11:
                logbook, name, fake_f, use_enc, keepbest, fake_m, config, fitness_len_weight, fitness_len_max_value, fitness_max_len, fitness_len_chars_sub = x
            elif len(x) == 12:
                logbook, name, fake_f, use_enc, keepbest, fake_m, config, fitness_len_weight, fitness_len_max_value, fitness_max_len, fitness_len_chars_sub, fitness_min_value = x
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
            if 'cells' in config:
                cells = config['cells']
            if 'tourney_size' in config:
                tsize = config['tourney_size']
            if 'POPSIZE' in config:
                POPSIZE = config['POPSIZE']
            if 'mut_magnitude' in config:
                mut_magnitude = config['mut_magnitude']
        try:

            nam_base = '_'.join(name.split("_"))
            nam_split = nam_base.split("_")
            suffix = ''

            if nam_split[0].startswith('f'):
                use_enc = False

            if use_enc and 'slp' not in use_operators:
                continue
            if not use_enc and 'ordinary' not in use_operators:
                continue

            if fitness_min_value != -1:
                continue

            if not use_native:
                if nam_split[0].startswith('f'):
                    continue

            if not nam_split[0].startswith('f') and config['test'] in untrained_test_numbers:
                suffix = 'untrained'

            if tsize not in included_t_sizes:
                continue

            if POPSIZE not in included_POPSIZES:
                continue

            if mut_magnitude not in included_mut_magnitude:
                continue

            if cells not in included_cells:
                continue

            if not use_untrained:
                if suffix == 'untrained':
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
            df_temp['num_cr'] = df_temp['gen'] * POPSIZE # POPULSIZE INSTEAD OF 50 should be there
            # df_temp = df_temp[df_temp['gen']<=100]
            df_temp['tsize'] = tsize

            df_temp['useEncoder'] = str(use_enc)


            # if 'nolatent' in nam_base:
            #     x=0
            # CMAES CUM MAX
            # df_temp = df_temp.sort_values(by='gen', ascending=True)
            # df_temp['best'] = df_temp['best'].cummax()
            gens_done = df_temp['gen'].max()
            if gens_done < required_generations:
                continue
            df_temp = df_temp[df_temp['gen'] <= max_generations]
            if only_best:
                df_temp = df_temp[df_temp['best'] == df_temp['best'].max()].iloc[0:1] # THIS LINE ENABLE/DISABLE
            # if
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

            df_temp['run'] = 0
            df_temp['mut_magnitude'] = mut_magnitude
            df_temp['cmaes'] = cmaes
            df_temp['cells'] = cells

            if custom_filter(df_temp):
                continue

            cnt[nam_base+"__"+suffix+"__" + str(tsize)+'_'+str(use_enc) + "__" + str(mut_magnitude) +"__"+str(cmaes)+"__" + str(fitness_min_value)] += 1
            # cnt[nam_base+str(config['mut_magnitude'])+str(tsize)] += 1

            print(name, df_temp['best'].max())
            print(len(df_temp))

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

df = df_latent

df = fix_dtypes(df)
df = df.sort_values(by='param')
plot_gens(df)
