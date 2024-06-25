import numpy as np
import torch
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
