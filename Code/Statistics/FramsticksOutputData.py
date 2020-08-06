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
    for fname in glob.glob('logbooks/**/logbook_*', recursive=True):
        dir, f = os.path.split(fname)
        dir, name = os.path.split(dir)
        num += 1
        print(fname)
        with open(fname, 'rb') as ff:
            logbook = pickle.load(ff)
        try:
            df_temp = pd.DataFrame(logbook).rename(columns={'max': 'best', 'min':'worst', 'nevals':'num_cr'})
            # filtering

            df_temp['num_cr'] = df_temp['gen'] * 50
            nam_base = '_'.join(name.split("_")[:-1])



            nam_split = nam_base.split("_")
            # if 'nolatent' in nam_base:
            #     x=0
            if nam_split[-1] == "Bidir":
                df_temp['param'] = 'lat'
                df_temp = df_temp[df_temp['gen'] == df_temp['gen'].max()]
            elif nam_split[-1] == "nolatent":
                df_temp['param'] = 'f1'
                df_temp = df_temp[df_temp['gen'] == df_temp['gen'].max()]
            elif nam_split[3] == "True":
                df_temp['param'] = 'f1_rand'
                df_temp = df_temp[df_temp['best'] == df_temp['best'].max()]
            elif nam_split[6] == "True":
                df_temp['param'] = 'lat_rand'
                df_temp = df_temp[df_temp['best'] == df_temp['best'].max()]
            else:
                nam_base=nam_base.replace("-", ".")
                nam_base=nam_base.replace("l", "L")
                nam_base=nam_base.replace("00", "0")
                df_temp = df_temp[df_temp['gen'] == df_temp['gen'].max()]
                df_temp['param'] = nam_base.split("_")[6]
            # nam_base = 'f1' if 'nolatent' in name else 'latent'
            # nam_base += '_random' if 'True' in name else ''
            cnt[nam_base] += 1

            df_temp['run'] = 0
            df = pd.concat([df, df_temp], axis=0)
        except:
            num -= 1
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


# plot_multiline_fitness(df)
plot_avg_fitness(df)
plot_best_fitness(df)
# plot_computation_time_for_parameterizations(df)