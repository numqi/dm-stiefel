import scipy.special
import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

import numqi

from utils import GeometricCoherenceModel, get_real_equal_prob_state_geometric_coherence

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


dim = 64
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

gc_analytical = get_real_equal_prob_state_geometric_coherence(dim, alpha_list)

fig,ax = plt.subplots()
ax.plot(alpha_list, gc_list, label='manifold-opt')
ax.plot(alpha_list, gc_analytical, 'x', label='analytical')
ax.legend()
ax.set_xlabel('alpha')
ax.set_ylabel('geometric coherence')
ax.set_title(f'dim={dim}')
fig.tight_layout()
fig.savefig('tbd01.png', dpi=200)
