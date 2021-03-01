import pickle
import os
import numpy as np
import re
import matplotlib.pyplot as plt

representation = 'f9'

plt.style.use(['science', 'no-latex', 'grid'])



def drawFDCs():
    with open('corrsFile', 'rb') as file:
        data = pickle.load(file)

    data = [(name, cr, cr_enc) for name, cr, cr_enc in data if representation in name]

    corrs = {}
    for name, cr, cr_enc in data:
        loc = name.split("_")[6]
        if 'f' in loc:
            nam = 'reconstruct + fitness'
        elif 'n' in loc:
            nam = 'reconstruct'
        else:
            nam = 'reconstruct + phenotype'
        corrs[nam] = cr
        corrs[nam + ' SLP'] = cr_enc

    fig = plt.figure(figsize=(6.4, 4.8))
    ax = fig.add_subplot(111)
    vals = np.arange(len(corrs))
    corr_values_points = [cr for cr in corrs.values()]
    points = list(zip(list(vals), corr_values_points))
    names = list(corrs.keys())
    names = [re.sub(' +', ' ', nam.replace('model', '')) for nam in names]
    for i in range(len(names)):
        plt.annotate(names[i], points[i], ha='center', va='bottom')
    ax.scatter(vals, list(corrs.values()), s=30)
    plt.ylabel("Fitness distance correlation", fontsize=12, labelpad=10)
    plt.xlabel("%s models" % representation, fontsize=12, labelpad=10)
    plt.tick_params(
        axis='x',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
        labelbottom=False)
    plt.xlim(-1, vals[-1] + 1)
    plt.ylim(-1, 1)
    plt.savefig(
        'corrsPlot'+representation+'.pdf',
        dpi=300,
        transparent=False, bbox_inches='tight')
    plt.show()
    plt.clf()


if __name__ == '__main__':
    drawFDCs()