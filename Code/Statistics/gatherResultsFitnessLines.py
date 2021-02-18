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

path_base = "mods2/mutateDistance/**/*calculateFitness*"

def gatherScores():

    dic = defaultdict(list)
    for f in glob.glob(path_base, recursive=True):
        with open(f, 'rb') as file:
            lines = pickle.load(file)



        dirr, filename = os.path.split(f)
        if not filename.startswith("model"):
            dirr, model_name = os.path.split(dirr)
            model_params = model_name.split('_')
        else:
            model_params = filename.split('_')[:-2]
        if model_params[0] == 'model':
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
        dic[locality].extend(lines)

    return dic
dic = gatherScores()

dicAll = defaultdict(list)
dicSelected = defaultdict(list)
for k,v in dic.items():

    for line in v:
        vals = line[1]
        vals = np.around(vals, 4)
        dif = np.diff(vals)
        grow_changes = np.where(dif>0,1,0)
        decay_changes = np.where(dif<0, 1, 0)

        joint_changes = grow_changes - decay_changes
        filtered_changes=joint_changes[joint_changes != 0]
        monotonic_changes = np.diff(filtered_changes)
        unit_monotonic_changes = np.where(monotonic_changes!=0, 1, 0)
        sum_monotonic_changes = np.sum(unit_monotonic_changes)

        sum_grow_changes = np.sum(grow_changes)
        sum_decay_changes = np.sum(decay_changes)
        all_changes = sum_grow_changes + sum_decay_changes
        dicAll[k].append(all_changes)
        dicSelected[k].append(sum_monotonic_changes)

colormap ={'noloc':'b',
           'levenshtein': 'orange',
           'fitness':'r',
           'dissim':'g'

}

namemap={'noloc':'none', 'levenshtein':'levenshtein', 'dissim':'phenotype', 'fitness':'fitness'}
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

reses = []
for key in dicAll.keys():


    # print(k)
#
    res=ax.scatter(dicAll[key], dicSelected[key] ,np.random.rand(len(dicSelected[key])), c=colormap[key], s=4, alpha=0.7)
    reses.append(res)
    allmean = np.mean(dicAll[key])
    selmean = np.mean(dicSelected[key])

    submean = np.mean(np.subtract(dicAll[key], dicSelected[key]))
    # divmean = np.mean(np.divide(dicAll[key], dicSelected[key]))
    print(namemap[key])
    print('all changes mean: ',allmean)
    print('changes in monotonicity mean: ',selmean)
    print('all divided monotonic changes mean: ', allmean/selmean)
    print('difference between all / monotonic changes mean: ', allmean-selmean)
    print('mean of sub at each point', submean)
    # print('mean of div at each point', divmean)
plt.xlabel('all changes')
plt.ylabel('changes in monotonicity')
plt.legend(reses, [namemap[key] for key in dicAll.keys()])
plt.show()
    # for i in range(100):
    #
    #     ax.plot(v[i][0], v[i][1])
    #
    # plt.show()
