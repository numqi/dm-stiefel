import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

import numqi

from utils import CoherenceFormationModel
from utils import GeometricCoherenceModel, get_maximally_coherent_state_mixed_coherence

def demo_CoherenceFormationModel():
    tmp0 = numqi.random.rand_haar_state(4)
    dm_target = tmp0.reshape(-1,1) * tmp0.conj()
    dim = dm_target.shape[0]
    alpha_list = np.linspace(0, 1, 50)

    model = CoherenceFormationModel(dim, num_term=3*dim)
    kwargs = dict(theta0='uniform', num_repeat=3, tol=1e-10, print_every_round=0)
    cof_list = []
    for alpha_i in tqdm(alpha_list):
        model.set_density_matrix(numqi.entangle.hf_interpolate_dm(dm_target, alpha=alpha_i))
        cof_list.append(numqi.optimize.minimize(model, **kwargs).fun)
    cof_list = np.array(cof_list)

    fig,ax = plt.subplots()
    ax.plot(alpha_list, cof_list)
    ax.set_xlabel('alpha')
    ax.set_ylabel('CoF')
    fig.tight_layout()
    fig.savefig('tbd00.png', dpi=200)


def demo_GeometricCoherenceModel():
    dim = 32 #when dim=64, more num_repeat is needed
    tmp0 = np.ones(dim, dtype=np.float64)/np.sqrt(dim)
    dm_target = tmp0.reshape(-1,1) * tmp0.conj()

    alpha_list = np.linspace(0, 1, 50)

    model = GeometricCoherenceModel(dim, num_term=4*dim, temperature=0.3)
    kwargs = dict(theta0='uniform', num_repeat=3, tol=1e-10, print_every_round=0)
    gc_list = []
    for alpha_i in tqdm(alpha_list):
        model.set_density_matrix(numqi.entangle.hf_interpolate_dm(dm_target, alpha=alpha_i))
        theta_optim = numqi.optimize.minimize(model, **kwargs).fun
        with torch.no_grad():
            gc_list.append(model(use_temperature=False).item())
    gc_list = np.array(gc_list)

    gc_analytical = get_maximally_coherent_state_mixed_coherence(dim, alpha_list)

    fig,ax = plt.subplots()
    ax.plot(alpha_list, gc_list, label='manifold-opt')
    ax.plot(alpha_list, gc_analytical, 'x', label='analytical')
    ax.legend()
    ax.set_xlabel('alpha')
    ax.set_ylabel('geometric coherence')
    ax.set_title(f'dim={dim}')
    fig.tight_layout()
    fig.savefig('tbd01.png', dpi=200)
