#!/usr/bin/env python
import sys
import os
from pathlib import Path
from time import sleep
def mkdir_p(dir):
    '''make a directory (dir) if it doesn't exist'''
    if not os.path.exists(dir):
        os.mkdir(dir)

onlyEval = 'None'
try:
    sys.argv[1]
except NameError:
    onlyEval = 'None'
else:
    if sys.argv[1] == 'eval':
        onlyEval = 'True'

#print(os.getcwd())
job_directory = "%s/job" % os.getcwd()
read_directory = "%s/data" % os.getcwd()
home = str(Path.home())
data_dir = os.path.join(home, 'dataFolder')



# Make top level directories
mkdir_p(job_directory)
mkdir_p(data_dir)


all_labs = ['44']*8 + ['43'] * 11  + ['45'] * 13 + ['ci'] * 13
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


options_representation = ['f1', 'f4', 'f9']
options_long_genos = ['None']
options_cells = ['16', '32', '64']
options_twoLayer = ['oneLayer', 'twoLayer']
options_bidir =['Singledir', 'Bidir']

i = 0
params = []
for rep in options_representation:
    for lon in options_long_genos:
        for cel in options_cells:
            for lay in options_twoLayer:
                for bid in options_bidir:
                    d = {}
                    d['model_name'] = 'model_%s_%s_%s_%s_%s' % (rep, lon, cel, lay, bid)
                    d['representation'] = rep
                    d['cells'] = cel
                    d['twoLayer'] = lay
                    d['bidir'] = bid
                    d['long_genos'] = lon
                    d['lab'] = all_labs[i]
                    d['node'] = nodes[i]
                    i+=1
                    params.append(d)

print("ok")
for param in params:


    model_name = param['model_name']+'_CHECK'
    lab = param['lab']
    representation = param['representation']
    cells = param['cells']
    twoLayer = param['twoLayer']
    bidir = param['bidir']
    long_genos = param['long_genos']
    node = param['node']
    # if not(representation == 'f9' and long_genos == 'long'):
    #     continue

    job_file = os.path.join(job_directory, "%s.job" % str(model_name))
    param_data = os.path.join(data_dir, str(model_name))

    outdir = "%s/out" % param_data

    # Create directories
    mkdir_p(param_data)
    mkdir_p(outdir)

    with open(job_file, 'w') as fh:
        fh.writelines("#!/bin/bash\n")
        fh.writelines("#SBATCH --job-name=%s.job\n" % model_name)
        fh.writelines("#SBATCH --time=11:59:59\n")
        fh.writelines("#SBATCH -p lab-%s-student\n" % lab)
        fh.writelines("#SBATCH --output=%s/%s.out\n" % (outdir,model_name))
        fh.writelines("#SBATCH --error=%s/%s.err\n" % (outdir,model_name))
        fh.writelines("#SBATCH --qos=normal\n")
        #fh.writelines("#SBATCH --nodelist=%s\n" % node)
        fh.writelines("#SBATCH --mincpus=4\n")
        fh.writelines("#SBATCH --mail-type=ALL\n")
        fh.writelines("#SBATCH --mail-user=piotr.kaszuba@student.put.poznan.pl\n")
        #fh.writelines("echo hello")
        fh.writelines("python3 $HOME/workspace/runFile.py %s %s %s %s %s %s %s %s %s\n" % (model_name, representation, long_genos, cells, twoLayer, bidir, param_data, read_directory, onlyEval))

    os.system("chmod 777 %s" % job_file)
    os.system("sbatch %s" % job_file)
    sleep(2)
    break
print("ok")