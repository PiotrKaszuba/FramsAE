import matplotlib.pyplot as plt
import numpy as np
import os
from collections import Counter
def plot(hist, acc, epochs, model_name, path):
    loss = [np.mean(his['loss'])  for his in hist]
    val = [np.mean(his['val_loss'])  for his in hist]

    plt.plot(epochs, acc)
    plt.title(model_name)
    plt.xlabel('epochs')
    plt.ylabel('accuracy every 50 epochs')
    plt.savefig(os.path.join(path,model_name+'_acc.png'))
    plt.clf()
    plt.plot(epochs, loss, 'b', label= 'loss')
    plt.plot(epochs, val, 'g', label= 'val_loss')
    plt.title(model_name)
    plt.xlabel('epochs')
    plt.ylabel('avg loss between 50 epochs')
    plt.legend()
    plt.savefig(os.path.join(path, model_name + '_loss.png'))
    plt.clf()

def plot_acc_by_length(acc, hits, zerosTestCheck, model_name, path):
    lens = list(map(np.sum, zerosTestCheck))
    min_len = min(lens)
    max_len = max(lens)

    cntLen = Counter()
    cntHits = Counter()

    for i in range(np.shape(zerosTestCheck)[0]):
        lenToAdd = zerosTestCheck[i].sum()
        cntLen[lenToAdd] += lenToAdd
        hitsToAdd = hits[i].sum()
        cntHits[lenToAdd] += hitsToAdd

    hitsAll = [cntHits[i]/cntLen[i] if cntLen[i] != 0 else 0.0  for i in range(min_len, max_len+1)]
    plt.bar(range(min_len, max_len+1), hitsAll, align='center', width=0.6)
    plt.xlabel('Geno length')
    plt.ylabel('% of tokens predicted right')
    plt.title(model_name+", overall acc: " + str(acc))
    plt.savefig(os.path.join(path, model_name + '_AccByLen.png'))
    plt.clf()

    # gg, cnts = np.unique(ff, return_counts=True)
    # plt.bar(range(min(ff), max(ff) + 1), cnts, align='center', width=0.6)
    # plt.show()

