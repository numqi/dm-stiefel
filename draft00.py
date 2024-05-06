import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

import numqi

from utils import CoherenceFormationModel


# tmp0 = numqi.state.maximally_entangled_state(2)
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
