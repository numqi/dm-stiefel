import os
import pickle
import numpy as np
import torch
import scipy.sparse
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt

import numqi

from utils import MagicStabilizerEntropyModel

if torch.get_num_threads()!=1:
    torch.set_num_threads(1)


def demo_H_state():
    num_qubit = 1
    alpha_list = [2,3,4]
    prob_list = np.linspace(0, 1, 100)
    Hstate = np.array([1, np.sqrt(2)-1]) / np.sqrt(4-2*np.sqrt(2))
    # psi_target = Hstate
    psi_target = numqi.random.rand_haar_state(2)
    dm_target = psi_target.reshape(-1,1) * psi_target.conj()
    alpha_boundary = 0.5 / np.abs(numqi.gellmann.dm_to_gellmann_basis(dm_target)).sum()

    ret_opt = []
    for alpha_i in alpha_list:
        model = MagicStabilizerEntropyModel(alpha_i, num_qubit, num_term=4*(2**num_qubit))
        for prob_i in tqdm(prob_list):
            model.set_density_matrix(numqi.entangle.hf_interpolate_dm(dm_target, alpha=prob_i))
            ret_opt.append(-numqi.optimize.minimize(model, 'uniform', num_repeat=10, tol=1e-10, print_every_round=0).fun)
    ret_opt = np.array(ret_opt).reshape(len(alpha_list), -1)

    fig,ax = plt.subplots()
    ax.axvline(alpha_boundary, linestyle=':', color='red', label=r'$0.5\|\vec{\rho}\|_1^{-1}$')
    for ind0 in range(len(alpha_list)):
        ax.plot(prob_list, 1-ret_opt[ind0], label=f'alpha={alpha_list[ind0]}')
    ax.set_xlabel(r'$p\rho + (1-p)I/d$')
    ax.set_ylabel('linear Stab Entropy')
    ax.set_xlim(0, 1)
    ax.set_title(f'random direction')
    ax.legend()
    ax.set_yscale('log')
    fig.tight_layout()
    fig.savefig('tbd02.png', dpi=200)
    # fig.savefig('data/linear_stab_entropy_H_state.png', dpi=200)


def demo_CS_state():
    alpha_list = [2]
    CSstate = np.array([1, 1, 1, 1j], dtype=np.complex128) / 2
    dm_target = CSstate.reshape(-1,1) * CSstate.conj()
    num_qubit = numqi.utils.hf_num_state_to_num_qubit(dm_target.shape[0])
    prob_list = np.linspace(0, 1, 50)

    ret_opt = []
    for alpha_i in alpha_list:
        model = MagicStabilizerEntropyModel(alpha_i, num_qubit, num_term=2*(2**num_qubit))
        for prob_i in tqdm(prob_list):
            model.set_density_matrix(numqi.entangle.hf_interpolate_dm(dm_target, alpha=prob_i))
            ret_opt.append(-numqi.optimize.minimize(model, 'uniform', num_repeat=3, tol=1e-10, print_every_round=0).fun)
            # numqi.optimize.minimize_adam(model, num_step=5000, theta0='uniform', optim_args=('adam', 0.03,0.01), tqdm_update_freq=0)
    ret_opt = np.array(ret_opt).reshape(len(alpha_list), -1)

    fig,ax = plt.subplots()
    ax.axvline(1/2, color='red', label='1/2')
    for ind0 in range(len(alpha_list)):
        ax.plot(prob_list, 1-ret_opt[ind0], label=f'alpha={alpha_list[ind0]}')
    ax.set_xlabel(r'$p\rho + (1-p)I/d$')
    ax.set_ylabel('linear Stab Entropy')
    ax.set_title(f'CS state')
    ax.legend()
    ax.set_yscale('log')
    fig.tight_layout()
    fig.savefig('tbd02.png', dpi=200)


def demo_Toffoli_state():
    alpha_list = [2]
    toffoli_state = np.zeros(8, dtype=np.float64)
    toffoli_state[[0,2,4,7]] = 1/2
    dm_target = toffoli_state.reshape(-1,1) * toffoli_state
    num_qubit = numqi.utils.hf_num_state_to_num_qubit(dm_target.shape[0])
    prob_list = np.linspace(0, 1, 50)

    ret_opt = []
    for alpha_i in alpha_list:
        model = MagicStabilizerEntropyModel(alpha_i, num_qubit, num_term=4*(2**num_qubit), method='polar')
        for prob_i in tqdm(prob_list):
            model.set_density_matrix(numqi.entangle.hf_interpolate_dm(dm_target, alpha=prob_i))
            ret_opt.append(-numqi.optimize.minimize(model, 'uniform', num_repeat=3, tol=1e-10, print_every_round=0).fun)
            # numqi.optimize.minimize_adam(model, num_step=5000, theta0='uniform', optim_args=('adam', 0.01, 0.001))
    ret_opt = np.array(ret_opt).reshape(len(alpha_list), -1)

    fig,ax = plt.subplots()
    ax.axvline(1/2, color='red', label='1/2')
    for ind0 in range(len(alpha_list)):
        ax.plot(prob_list, 1-ret_opt[ind0], label=f'alpha={alpha_list[ind0]}')
    ax.set_xlabel(r'$p\rho + (1-p)I/d$')
    ax.set_ylabel('linear Stab Entropy')
    ax.set_title(f'Toffoli state')
    ax.legend()
    ax.set_yscale('log')
    fig.tight_layout()
    fig.savefig('tbd02.png', dpi=200)


def demo_logsumexp():
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


def demo_bloch_cross_section():
    datapath = 'data/bloch_cross_section.pkl'
    xlist = np.linspace(-0.54, 0.54, 101)
    ylist = np.linspace(-0.54, 0.54, 101)

    if not os.path.exists(datapath):
        model = MagicStabilizerEntropyModel(alpha=2, num_qubit=1, num_term=4)
        ret_opt = np.zeros((len(xlist)*len(ylist)), dtype=np.float64)
        # to completely remove those coarse points, a larger num_repeat is needed
        kwargs = dict(theta0='uniform', num_repeat=100, tol=1e-10, print_every_round=0, early_stop_threshold=(1e-8)-1)
        tmp0 = tuple((xi,yi) for xi in xlist for yi in ylist)
        for ind0,(xi,yi) in tqdm(enumerate(tmp0), total=len(xlist)*len(ylist)):
            tmp0 = np.array([xi, yi, 0])
            if np.linalg.norm(tmp0) > 0.5:
                ret_opt[ind0] = np.nan
            else:
                model.set_density_matrix(numqi.gellmann.gellmann_basis_to_dm(tmp0))
                ret_opt[ind0] = 1-(-numqi.optimize.minimize(model, **kwargs).fun)
        ret_opt = ret_opt.reshape(len(xlist), len(ylist))
        with open(datapath, 'wb') as fid:
            pickle.dump(dict(xlist=xlist, ylist=ylist, ret_opt=ret_opt), fid)
    else:
        with open(datapath, 'rb') as fid:
            tmp0 = pickle.load(fid)
            xlist = tmp0['xlist']
            ylist = tmp0['ylist']
            ret_opt = tmp0['ret_opt']

    tmp0 = ret_opt[np.logical_not(np.isnan(ret_opt))]
    ret_opt_min = tmp0.min()
    ret_opt_max = tmp0.max()
    print(ret_opt_min, ret_opt_max)
    # plt.get_cmap('winter')(np.array([0.0, 1.0])) #RGBA

    z0 = ret_opt.copy()
    z0[np.isnan(z0)] = ret_opt_min
    z0 = np.clip(z0, 1e-7, z0.max())
    tmp0 = (xlist.reshape(-1,1)**2 + ylist.reshape(1,-1)**2) > 0.26
    z0[tmp0] = np.nan
    fig,ax = plt.subplots()
    hcontourf = ax.contourf(xlist, ylist, np.log10(z0.T), levels=30, cmap='RdPu')
    tmp0 = np.linspace(0, 2*np.pi, 100)
    ax.plot(0.5*np.cos(tmp0), 0.5*np.sin(tmp0), linestyle='solid', color='black', linewidth=3)
    ax.set_aspect('equal')
    cax = fig.colorbar(hcontourf, shrink=0.8)
    cax.ax.get_yticks()
    tmp0 = list(range(-6, 0))
    cax.ax.set_yticks(tmp0)
    cax.ax.set_yticklabels(['$10^{}$'.format('{'+str(x)+'}') for x in tmp0])
    ax.axis('off')
    fig.tight_layout()
    fig.savefig('tbd02.png', dpi=200)


if __name__=='__main__':
    demo_bloch_cross_section()
