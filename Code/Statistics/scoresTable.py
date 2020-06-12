import pandas as pd
import glob
import os
import pickle
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
ppath = "tempDataFolders/"
out = "scores_table_new.csv"
parent, subFolders, _ = next( os.walk(ppath))
print(subFolders)

mean_examples = 10
allSeries = []
for sub in subFolders:
    split = sub.split("_")
    if len(split) != 6:
        continue
    modStr, repres, non, cells, layers, directions = split



    path = os.path.join(ppath, sub, sub+"_losses")


    assert( os.path.exists(path))



    with open(path, 'rb') as handle:
        losses_file = pickle.load(handle)

    hist = losses_file['history']
    accs = losses_file['accuracy']
    eps_tot = losses_file['epochs_total']


    epse=eps_tot[-1]

    if len(accs) > mean_examples:
        avg= np.mean(accs[-mean_examples:])
    else:
        avg = np.mean(accs)
    series = pd.Series([sub, repres, cells, layers, directions, epse, avg])
    allSeries.append(series)



fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

df=pd.concat(allSeries, axis=1)
df=df.transpose()
df.columns = ['model', 'representation', 'cells', 'layers', 'directions', "total_epochs", 'smoothed_acc']

dirEncoder = LabelEncoder()
df['directions'] = dirEncoder.fit_transform(df['directions'])

cellEncoder = LabelEncoder()
df['cells'] = cellEncoder.fit_transform(df['cells'])

# reprEncoder = LabelEncoder()
# df['representation']=reprEncoder.fit_transform(df['representation'])

layerEncoder = LabelEncoder()

df['layers'] = layerEncoder.fit_transform(df['layers'])
offset = lambda p: transforms.ScaledTranslation(p/72.,0, plt.gcf().dpi_scale_trans)
trans = plt.gca().transData

f1Df = df[df['representation'] == 'f1']
f4Df = df[df['representation'] == 'f4']
f9Df = df[df['representation'] == 'f9']
f1=ax.scatter(f1Df['directions'], f1Df['cells'], f1Df['layers'], s=f1Df['smoothed_acc'].astype(np.float64)*120, transform=trans+offset(-12), alpha=1.0)
f4=ax.scatter(f4Df['directions'], f4Df['cells'], f4Df['layers'],s=f4Df['smoothed_acc'].astype(np.float64)*120, transform=trans+offset(0), alpha=1.0)
f9=ax.scatter(f9Df['directions'], f9Df['cells'], f9Df['layers'],s=f9Df['smoothed_acc'].astype(np.float64)*120, transform=trans+offset(+12), alpha=1.0)
plt.legend((f1,f4,f9),
           ('f1', 'f4', 'f9'),
           scatterpoints=1,
           loc='lower left',
           ncol=3,
           fontsize=8)

ax.set_xticks(np.arange(3))
ax.set_xticklabels(dirEncoder.classes_)

ax.set_yticks(np.arange(3))
ax.set_yticklabels(cellEncoder.classes_)

ax.set_zticks(np.arange(2))
ax.set_zticklabels(layerEncoder.classes_)
#plt.xticklabels(["3", "3fe", "wfrf"])
#fig.colorbar(img)
plt.show()
df.to_csv(out, index=False)