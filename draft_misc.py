import scipy.special
import numpy as np

import matplotlib.pyplot as plt

def demo_logsumexp():
    xdata = np.linspace(-2, 2, 301)
    T_list = [0.01, 0.03, 0.1, 0.3, 1]

    tmp0 = np.stack([xdata, -xdata], axis=1)
    ydata_list = np.stack([x*scipy.special.logsumexp(tmp0/x, axis=1) for x in T_list])

    fig,ax = plt.subplots()
    for ind0 in reversed(range(len(T_list))):
        ax.plot(xdata, ydata_list[ind0], label=f'T={T_list[ind0]}')
    ax.plot(xdata, np.maximum(xdata,-xdata), label='maximum')
    ax.legend()
    fig.tight_layout()
    fig.savefig('tbd02.png', dpi=200)


def demo_logsumexp01():
    xdata = np.linspace(-0.01, 1, 301)
    hf0 = lambda x: np.maximum(x, 1e-10)
    T_list = [0.001,0.01, 0.03, 0.1, 0.3]

    tmp0 = np.stack([xdata, hf0(xdata)], axis=1)
    ydata_list = np.stack([x*scipy.special.logsumexp(tmp0/x, axis=1) for x in T_list])

    fig,ax = plt.subplots()
    for ind0 in reversed(range(len(T_list))):
        ax.plot(xdata, ydata_list[ind0]-hf0(xdata), label=f'T={T_list[ind0]}')
    # ax.plot(xdata, hf0(xdata), label='maximum')
    ax.set_yscale('log')
    ax.legend()
    fig.tight_layout()
    fig.savefig('tbd02.png', dpi=200)
