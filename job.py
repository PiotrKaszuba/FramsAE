#!/usr/bin/env python
import sys
import os
from pathlib import Path
from time import sleep
import numpy as np
def mkdir_p(dir):
    '''make a directory (dir) if it doesn't exist'''
    if not os.path.exists(dir):
        os.mkdir(dir)

# 'True' / 'False' options:
clear_files = 'True'

# important options:
options_representation = ['']
options_long_genos = ['None']
options_cells = ['64']#, '56', '48']
options_twoLayer = ['twoLayer']
options_bidir = ['Bidir']
options_locality = [""]
options_tests = ['19'] #list(range(1))
generations = '250'
population_size = '50'
tourney_size = '3' # 1 equals random search
task_evol_test_no = list(range(1))
# task_evol_test_no = list((np.arange(1) + 1))
# available options:
tasks = ['train', 'evol', 'collect_data', 'evaluate_model', 'FDC']
types_of_run = ['locality_train', 'noloc_train', 'evol_autoencoders', 'evol_random', 'mutDist_latent']
latents = ['nolatent', 'latent']

task = tasks[1]
type_of_run = types_of_run[4]
latent = latents[1]
# MAIN BRANCHES:
if type_of_run == "locality_train":
    options_locality = ["0-0n", "3", "3f"]

    # options_locality = ["0-1", "1", "10", "0-33", "3", "0-1l", "1l", "10l", "0-33l", "3l", "0-0n", "0-00n", "0", "0n"]
    # options_tests = list(range(3))
elif type_of_run == 'noloc_train':
    #options_representation = ['f1', 'f4', 'f9']
    options_cells = ['16', '32', '64']
    options_twoLayer = ['oneLayer', 'twoLayer']
    options_bidir = ['Singledir', 'Bidir']
elif type_of_run == 'evol_autoencoders':
    pass
    # options_locality = ["3", "3f"]
    # options_locality = ['0-0n', '3', '3f']#["3", "3l", "0n", "3f"]
    # options_tests = [1, 2]
    # task_evol_test_no = list((np.arange(1) + 1))

elif type_of_run == 'evol_random':
    options_locality = ['0-0n']
    # tourney_size = '1'
    task_evol_test_no = list((np.arange(2)+1))
elif type_of_run == 'mutDist_latent':
    latent = latents[1]
    task_evol_test_no = list((np.arange(15) + 11))
# elif type_of_run == 'collect_data'




# LAB settings
all_labs = ['ci'] * 2 + ['43'] * 2  + ['45'] * 2  + ['44'] * 2
all_labs = all_labs * 1000
# nodes = []
# Nodes settings
# for i in range(4): #up to 6
#     nodes.append("lab-43-[%d]" % (i+1))
# for i in range(7):
#     nodes.append("lab-43-[%d]" % (i+8))
#
# for i in range(12):
#     nodes.append("lab-45-[%d]" % (i + 3))
# for i in range(12):
#     nodes.append("lab-ci-[%d]" % (i + 3))
#
# for i in range(2):
#     nodes.append("lab-44-[%d]" % (i + 1))
# for i in range(4):
#     nodes.append("lab-44-[%d]" % (i + 5))
# for i in range(3):
#     nodes.append("lab-44-[%d]" % (i + 14))





#print(os.getcwd())

# directories setup
job_directory = "%s/job" % os.getcwd()
read_directory = "%s/data" % os.getcwd()
home = str(Path.home())
data_dir = os.path.join(home, 'dataFolder2')
frams_path = os.path.join(home, 'Framsticks50rc17')
# Make top level directories
mkdir_p(job_directory)
mkdir_p(data_dir)



additional_options_dict = [
    # {'placeholder':13, 'latent':'latent'},
    # {'placeholder':17, 'latent':'nolatent'},
    # {'placeholder':19, 'latent':'latent', 'test':16},
    # {'placeholder':23, 'latent':'latent', 'tourney_size':3},
    # {'placeholder':29, 'latent':'nolatent', 'tourney_size':3},
    # {'placeholder':31, 'latent':'latent', 'test':16, 'tourney_size':3},

    # {'placeholder':37, 'latent':'nolatent', 'representation':'f1'},
    # {'placeholder':41, 'latent':'nolatent', 'representation':'f1', 'tourney_size':3}
    # {'placeholder':43, 'locality':'0-0n', 'test':16},
    # {'placeholder':47, 'locality':'0-0n', 'test':16, 'representation':'f1'},
    # {'placeholder':53, 'locality':'3', 'test':16},
    # {'placeholder':59, 'locality':'3', 'test':16, 'representation':'f1'},
    # {'placeholder':61, 'latent':'nolatent'},
    # {'placeholder': 67, 'latent': 'nolatent', 'representation': 'f1'},
    # {'placeholder':71, 'locality':'3f', 'test':16},
    # {'placeholder':73, 'locality':'3f', 'test':16, 'representation':'f1'},
    # {'placeholder':79*7, 'locality':'0-0n', 'test':16},
    # {'placeholder':83*7, 'locality':'3', 'test':16},
    # {'placeholder':89*7, 'locality':'3f', 'test':16},

    # {'placeholder':97*131, 'locality':'0-0n', 'test':16, 'representation':'f9'}, #NOT RUN
    # {'placeholder':101*131, 'locality':'3', 'test':16, 'representation':'f9'}, # NOT RUN
    # {'placeholder':103*131, 'locality':'3f', 'test':16, 'representation':'f9'}, # TO CHANGE
    # {'placeholder': 109*131, 'locality': '0-0n', 'test': 16, 'representation': 'f1'},  # NOT RUN
    # {'placeholder': 113*131, 'locality': '3', 'test': 16, 'representation': 'f1'},  # NOT RUN
    # {'placeholder': 127*131, 'locality': '3f', 'test': 16, 'representation': 'f1'},  # TO CHANGE

    # {'placeholder':191*7, 'locality':'0-0n', 'test':16, 'representation': 'f1'},
    # {'placeholder':193*7, 'locality':'3', 'test':16, 'representation': 'f1'},
    # {'placeholder':197*7, 'locality':'3f', 'test':16, 'representation': 'f1'},
    # {'placeholder':19*7*163, 'latent':'latent', 'locality':'0-0n', 'test':16, 'representation':'f1'},
    # {'placeholder':19*7*167, 'latent':'latent', 'locality':'3', 'test':16, 'representation':'f1'},
    # {'placeholder': 19*7*173, 'latent': 'latent', 'locality':'3f',  'test': 16, 'representation': 'f1'},
    # {'placeholder':19*179, 'latent':'nolatent', 'representation': 'f1', 'locality':'0-0n', 'test':99},


    # {'placeholder':61*251, 'latent':'nolatent', 'representation': 'f9'},
    # {'placeholder': 67*251, 'latent': 'nolatent', 'representation': 'f1'},
    # for
    # {'placeholder':809, 'test':18},
    # {'placeholder':811, 'test':19},
    # {'placeholder':821, 'test':20},
    # {'placeholder' : 823},
    {'placeholder':863, 'latent':'nolatent', 'representation': 'f9' , 'locality': '0-0n', 'test':999,},
    {'placeholder':877, 'latent': 'nolatent', 'representation': 'f1' , 'locality': '0-0n', 'test':999,},

    {'placeholder' : 827, 'latent' : 'latent', 'locality': '0-0n', 'test':19, 'representation':'f1',},
    {'placeholder' : 829, 'latent' : 'latent', 'locality': '3', 'test':19, 'representation':'f1',},
    {'placeholder' : 839, 'latent' : 'latent', 'locality': '3f', 'test':19, 'representation':'f1',},
    {'placeholder' : 853, 'latent' : 'latent', 'locality': '0-0n', 'test':19, 'representation':'f9',},
    {'placeholder' : 857, 'latent' : 'latent', 'locality': '3', 'test':19, 'representation':'f9',},
    {'placeholder' : 859, 'latent' : 'latent', 'locality': '3f', 'test':19, 'representation':'f9',},



]

i = 0
params = []
for rep in options_representation:
    for lon in options_long_genos:
        for cel in options_cells:
            for lay in options_twoLayer:
                for bid in options_bidir:
                    for loc in options_locality:
                        for tes in options_tests:
                            for task_test in task_evol_test_no:
                                for a_dict in additional_options_dict:
                                    d = {}


                                    d['representation'] = rep
                                    d['cells'] = cel
                                    d['twoLayer'] = lay
                                    d['bidir'] = bid
                                    d['long_genos'] = lon
                                    d['lab'] = all_labs[i]
                                    # d['node'] = nodes[i]
                                    d['locality'] = loc
                                    d['test'] = tes
                                    d['task_test'] = task_test
                                    d.update(a_dict)
                                    d['model_name'] = 'model_%s_%s_%s_%s_%s_%s_%s' % (d['representation'], d['long_genos'], d['cells'], d['twoLayer'], d['bidir'], d['locality'], d['test'])
                                    i+=1
                                    params.append(d)

print("ok")
for param in params:


    model_name = param['model_name']
    lab = param['lab']
    representation = param['representation']
    cells = param['cells']
    twoLayer = param['twoLayer']
    bidir = param['bidir']
    long_genos = param['long_genos']
    # node = param['node']
    loc = param['locality']
    tes = param['test']
    task_tes = param['task_test']
    # if not(representation == 'f9' and long_genos == 'long'):
    #     continue
    placeholder = param['placeholder']
    task_tes = task_tes * placeholder
    instance_name = "%s_%s" % (str(model_name), str(task_tes))
    jobname = "%s.job" % instance_name
    job_file = os.path.join(job_directory, jobname)
    param_data = os.path.join(data_dir, str(model_name))

    outdir = "%s/out" % param_data

    # Create directories
    mkdir_p(param_data)
    mkdir_p(outdir)
    lat = latent if 'latent' not in param else param['latent']
    trn_siz = tourney_size if 'tourney_size' not in param else param['tourney_size']
    with open(job_file, 'w') as fh:
        fh.writelines("#!/bin/bash\n")
        fh.writelines("#SBATCH --job-name=%s\n" % jobname)
        fh.writelines("#SBATCH --time=23:59:59\n")
        fh.writelines("#SBATCH -p lab-%s-student\n" % lab)
        fh.writelines("#SBATCH --output=%s/%s.out\n" % (outdir,instance_name))
        fh.writelines("#SBATCH --error=%s/%s.err\n" % (outdir,instance_name))
        fh.writelines("#SBATCH --qos=normal\n")
        #fh.writelines("#SBATCH --nodelist=%s\n" % node)
        fh.writelines("#SBATCH --mincpus=2\n")
        fh.writelines("#SBATCH --mail-type=ALL\n")
        # fh.writelines("#SBATCH --mail-user=piotr.kaszuba@student.put.poznan.pl\n")
        #fh.writelines("echo hello")
        fh.writelines("stdbuf -o0 -e0 python3 $HOME/workspace/runFile.py %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s\n" % (model_name, representation, long_genos, cells, twoLayer, bidir, param_data, read_directory, task, loc, tes, frams_path, clear_files, lat, trn_siz, generations, population_size, task_tes))

    os.system("chmod 777 %s" % job_file)
    os.system("stdbuf -o0 -e0 sbatch %s" % job_file)
    sleep(0.2)
print("ok")