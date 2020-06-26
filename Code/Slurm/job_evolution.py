#!/usr/bin/env python
import sys
import os
from pathlib import Path
from time import sleep
def mkdir_p(dir):
    '''make a directory (dir) if it doesn't exist'''
    if not os.path.exists(dir):
        os.mkdir(dir)



#print(os.getcwd())
job_directory = "%s/job" % os.getcwd()
read_directory = "%s/data" % os.getcwd()
home = str(Path.home())
data_dir = os.path.join(home, 'dataFolder')

frams_path = os.path.join(home, 'Framsticks50rc14')

# Make top level directories
mkdir_p(job_directory)
mkdir_p(data_dir)


all_labs = ['43'] * 11  + ['45'] * 13 + ['ci'] * 13 + ['44']*8
nodes = []

for i in range(4): #up to 6
    nodes.append("lab-43-[%d]" % (i+1))
for i in range(7):
    nodes.append("lab-43-[%d]" % (i+8))

for i in range(12):
    nodes.append("lab-45-[%d]" % (i + 3))
for i in range(12):
    nodes.append("lab-ci-[%d]" % (i + 3))

for i in range(2):
    nodes.append("lab-44-[%d]" % (i + 1))
for i in range(4):
    nodes.append("lab-44-[%d]" % (i + 5))
for i in range(3):
    nodes.append("lab-44-[%d]" % (i + 14))


options_representation = ['f1']#, 'f4', 'f9']

latent = ['nolatent', 'model_64_oneLayer_Bidir']#, 'model_64_oneLayer_Bidir']


test_no = list(range(10))


i = 0
params = []
for rep in options_representation:
    for lat in latent:
        for test_n in test_no:
                    d = {}
                    d['evolution_name'] = 'evol_%s_%s_%s' % (rep, lat, str(test_n))
                    d['representation'] = rep
                    d['latent'] = lat
                    # d['test_n'] = test_n
                    d['lab'] = all_labs[i]
                    d['node'] = nodes[i]
                    i+=1
                    params.append(d)

print("ok")
for param in params:


    evolution_name = param['evolution_name']
    lab = param['lab']
    representation = param['representation']
    lat = param['latent']
    node = param['node']
    # if not(representation == 'f9' and long_genos == 'long'):
    #     continue

    job_file = os.path.join(job_directory, "%s.job" % str(evolution_name))
    param_data = os.path.join(data_dir, str(evolution_name))

    outdir = "%s/out" % param_data

    # Create directories
    mkdir_p(param_data)
    mkdir_p(outdir)

    with open(job_file, 'w') as fh:
        fh.writelines("#!/bin/bash\n")
        fh.writelines("#SBATCH --job-name=%s.job\n" % evolution_name)
        fh.writelines("#SBATCH --time=23:59:59\n")
        fh.writelines("#SBATCH -p lab-%s-student\n" % lab)
        fh.writelines("#SBATCH --output=%s/%s.out\n" % (outdir, evolution_name))
        fh.writelines("#SBATCH --error=%s/%s.err\n" % (outdir, evolution_name))
        fh.writelines("#SBATCH --qos=normal\n")
        #fh.writelines("#SBATCH --nodelist=%s\n" % node)
        # fh.writelines("#SBATCH --mincpus=1\n")
        fh.writelines("#SBATCH --mail-type=ALL\n")
        fh.writelines("#SBATCH --mail-user=piotr.kaszuba@student.put.poznan.pl\n")
        #fh.writelines("echo hello")
        fh.writelines("python3 $HOME/workspace/runFileEvol.py %s %s %s %s %s %s\n" % (evolution_name, representation, lat, param_data, read_directory, frams_path))

    os.system("chmod 777 %s" % job_file)
    os.system("sbatch %s" % job_file)
    sleep(2)
print("ok")