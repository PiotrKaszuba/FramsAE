import pandas as pd
import glob
import os
import pickle
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
ppath = "mods2/"
out = "scores_table_new.csv"
parent, subFolders, _ = next( os.walk(ppath))
print(subFolders)

mean_examples = 5
allSeries = []
for loss_file in glob.glob(ppath+"*_losses"):
# for sub in subFolders:
    dir, sub =os.path.split(loss_file)
    print(sub)

    split = sub.split("_")
    # if len(split) != 6:
    #     continue
    modStr, repres, non, cells, layers, directions, locality, test, _ = split



    # path = os.path.join(ppath, sub, sub+"_losses")


    assert( os.path.exists(loss_file))



    with open(loss_file, 'rb') as handle:
        losses_file = pickle.load(handle)

    hist = losses_file['history']
    accs = np.array(losses_file['accuracy'])
    eps_tot = np.array(losses_file['epochs_total'])


    # epse=eps_tot[-1]
    # print(epse)

    # if len(accs) > mean_examples:
    #     avg= np.mean(accs[-mean_examples:])
    # else:
    #     avg = np.mean(accs)
    max_ep = eps_tot[-1]
    # max_ep = min(np.max(eps_tot), 500)
    # if max_ep < 500:
    #     max_ep = 250


    inds = np.where((eps_tot <= max_ep) & (eps_tot>= max_ep-mean_examples+1))
    if len(inds[0]) == 0:
        continue
    print(inds)
    epse = eps_tot[inds]
    med = np.max(accs[inds])
    last_eps_locality = np.array(hist)[inds[0]]
    s = 0.0
    c = 0
    for h in last_eps_locality:
        for loc in h['locality']:
            if loc != 0.0:
                s += loc
            c+=1
    loc_avg = s/c
    corr = 1-loc_avg/3
    series = pd.Series([sub, repres, cells, layers, directions, epse, med, non, test, max_ep, locality, loc_avg, corr])
    allSeries.append(series)



fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

df=pd.concat(allSeries, axis=1)
df=df.transpose()
df.columns = ['model', 'representation', 'cells', 'layers', 'directions', "total_epochs", 'smoothed_acc', 'modif', 'version', 'epochs','locality', 'locality_avg', 'loc_corr']

# dirEncoder = LabelEncoder()
# df['directions'] = dirEncoder.fit_transform(df['directions'])

# cellEncoder = LabelEncoder()
# df['modif'] = cellEncoder.fit_transform(df['modif'])

# reprEncoder = LabelEncoder()
# df['representation']=reprEncoder.fit_transform(df['representation'])

# layerEncoder = LabelEncoder()

# df['layers'] = layerEncoder.fit_transform(df['layers'])
offset = lambda p: transforms.ScaledTranslation(p/72.,0, plt.gcf().dpi_scale_trans)
trans = plt.gca().transData
print(df[['loc_corr','representation', 'layers', 'directions', 'modif', 'version', 'epochs', 'smoothed_acc', 'representation','locality', 'locality_avg']])
f1Df = df[df['representation'] == 'f1']
f4Df = df[df['representation'] == 'f4']
f9Df = df[df['representation'] == 'f9']
f1=ax.scatter(f1Df['directions'], f1Df['modif'], f1Df['layers'], s=f1Df['smoothed_acc'].astype(np.float64)*120, transform=trans+offset(-12))
# f4=ax.scatter(f4Df['directions'], f4Df['cells'], f4Df['layers'],s=f4Df['smoothed_acc'].astype(np.float64)*120, transform=trans+offset(0))
f9=ax.scatter(f9Df['directions'], f9Df['modif'], f9Df['layers'],s=f9Df['smoothed_acc'].astype(np.float64)*120, transform=trans+offset(+12))
plt.legend((f1,f9),
           ('f1', 'f9'),
           scatterpoints=1,
           loc='lower right',
           ncol=3,
           fontsize=8)
ax.set_title("Relative accuracy of LSTM Autoencoders (radius of dot)")
ax.set_xlabel("Encoder LSTM mode")
ax.set_zlabel("LSTM layers")
ax.set_ylabel("LSTM cells")
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