import matplotlib.pyplot as plt
import numpy as np

import matplotlib as mpl

plt.style.use(['science', 'no-latex', 'grid'])

c1 = '#1f77b4'
def plot_f(ax, xlabel, ylabel, fontsize, name, legend=True, draw_minor=True):
    if not draw_minor:
        ax.xaxis.set_tick_params(which='minor', size=0, width=1, direction='in', top='on')
        ax.yaxis.set_tick_params(which='minor', size=0, width=1, direction='in', right='on')

    # ax.xaxis.set_tick_params(which='major', size=10, width=1, direction='in', top='on')
    # ax.xaxis.set_tick_params(which='minor', size=6, width=1, direction='in', top='on')
    # ax.yaxis.set_tick_params(which='major', size=10, width=1, direction='in', right='on')
    # ax.yaxis.set_tick_params(which='minor', size=6, width=1, direction='in', right='on')
    ax.set_xlabel(xlabel, fontsize=fontsize, labelpad=10)
    ax.set_ylabel(ylabel, fontsize=fontsize, labelpad=10)

    # ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(30))
    # ax.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(10))
    # ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(0.2))
    # ax.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(0.1))
    if legend:
        ax.legend(frameon=False, fontsize=fontsize)
    plt.savefig('C:\\Users\\Piotr\\Desktop\\Magisterka\\praca mag\\obrazki praca\\auto_generated\\%s.png'%name, dpi=300,
                transparent=False, bbox_inches='tight')
    plt.show()
    plt.clf()
def exponential_distribution(min, max, a, b, base):
    # base**(ax+1)+b
    lengths = np.arange(min, max+1)
    probs = base**(a*lengths+1) + b
    probs = probs / np.sum(probs)
    return probs
if __name__ == '__main__':
    fig = plt.figure(figsize=(6.4, 4.8))
    ax = plt.axes()
    # plt.axes(rect, projection=None, polar=False, **kwargs)
    plt.axes(ax)


    f1 = exponential_distribution(1,20, 0.25, -2, 2)
    f9 = exponential_distribution(1,20, 0.15, -2, 2)
    plt.bar(np.arange(20)+1, f9, alpha=1, label='f9', color='orange', fill=True)
    plt.bar(np.arange(20)+1, f1, alpha=1, label='f1', edgecolor='blue', fill=False)
    plot_f(ax, "Genotype length", "Probability", 12, "length_distribution")
    # plt.show()