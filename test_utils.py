import numpy as np
import torch

import numqi

from utils import (get_coherence_of_formation_1qubit, get_coherence_of_formation_pure, CoherenceFormationModel,
                GeometricCoherenceModel, get_geometric_coherence_1qubit, get_geometric_coherence_pure,
                get_real_equal_prob_state_geometric_coherence)


def test_CoherenceFormationModel_1qubit():
    dim = 2
    model = CoherenceFormationModel(dim, num_term=3*dim)
    for _ in range(10):
        rho = numqi.random.rand_density_matrix(2)
        ret_ = get_coherence_of_formation_1qubit(rho)
        model.set_density_matrix(rho)
        theta_optim = numqi.optimize.minimize(model, 'uniform', num_repeat=3, tol=1e-10, print_every_round=0)
        assert abs(ret_-theta_optim.fun) < 1e-7


def test_get_coherence_of_formation_1qubit():
    for _ in range(3):
        psi = numqi.random.rand_haar_state(2)
        ret_ = get_coherence_of_formation_pure(psi)
        tmp0 = psi.reshape(-1,1) * psi.conj()
        ret0 = get_coherence_of_formation_1qubit(tmp0)
        assert abs(ret_-ret0) < 1e-10

def test_GeometricCoherenceModel():
    dim = 2
    model = GeometricCoherenceModel(dim, num_term=4*dim, temperature=0.3)
    for _ in range(10):
        rho = numqi.random.rand_density_matrix(2)
        ret_ = get_geometric_coherence_1qubit(rho)
        model.set_density_matrix(rho)
        theta_optim = numqi.optimize.minimize(model, 'uniform', num_repeat=3, tol=1e-10, print_every_round=0)
        with torch.no_grad():
            ret0 = model(use_temperature=False).item()
        assert abs(ret_-ret0) < 1e-7


def test_get_geometric_coherence_1qubit():
    for _ in range(3):
        psi = numqi.random.rand_haar_state(2)
        ret_ = get_geometric_coherence_pure(psi)
        tmp0 = psi.reshape(-1,1) * psi.conj()
        ret0 = get_geometric_coherence_1qubit(tmp0)
        assert abs(ret_-ret0) < 1e-10


def test_get_real_equal_prob_state_geometric_coherence():
    dim = 3
    tmp0 = np.ones(dim, dtype=np.float64)/np.sqrt(dim)
    dm_target = tmp0.reshape(-1,1) * tmp0.conj()
    alpha_list = np.linspace(0, 1, 10)

    model = GeometricCoherenceModel(dim, num_term=4*dim, temperature=0.3)
    kwargs = dict(theta0='uniform', num_repeat=3, tol=1e-10, print_every_round=0)
    gc_list = []
    for alpha_i in alpha_list:
        model.set_density_matrix(numqi.entangle.hf_interpolate_dm(dm_target, alpha=alpha_i))
        theta_optim = numqi.optimize.minimize(model, **kwargs).fun
        with torch.no_grad():
            gc_list.append(model(use_temperature=False).item())
    gc_list = np.array(gc_list)
    gc_analytical = get_real_equal_prob_state_geometric_coherence(dim, alpha_list)
    assert np.abs(gc_list-gc_analytical).max() < 1e-7
