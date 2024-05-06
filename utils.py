import numpy as np
import torch

import numqi

def get_coherence_of_formation_pure(psi:np.ndarray) -> float:
    assert psi.ndim==1
    tmp0 = psi.real*psi.real + psi.imag*psi.imag
    eps = np.finfo(tmp0.dtype).eps
    tmp1 = np.log(np.maximum(tmp0, eps))
    ret = -np.dot(tmp0, tmp1)
    return ret


def get_coherence_of_formation_1qubit(rho:np.ndarray) -> float:
    assert rho.shape == (2,2)
    tmp0 = 0.5 + 0.5*np.sqrt(max(0, 1-4*abs(rho[0,1])**2))
    ret = float(-tmp0*np.log(tmp0) - (1-tmp0)*np.log(1-tmp0))
    return ret


class CoherenceFormationModel(torch.nn.Module):
    def __init__(self, dim:int, num_term:int, rank:int|None=None):
        super().__init__()
        self.dtype = torch.float64
        self.cdtype = torch.complex128
        self.dim = int(dim)
        self.num_term = int(num_term)
        if rank is None:
            rank = self.dim
        assert num_term>=rank
        assert rank!=1, 'for pure state, call "numqi.entangle.get_coherence_of_formation_pure()" instead'
        self.manifold = numqi.manifold.Stiefel(num_term, rank, dtype=self.cdtype, method='polar')
        self.rank = rank

        self._sqrt_rho_Tconj = None
        self._eps = torch.tensor(torch.finfo(self.dtype).smallest_normal, dtype=self.dtype)

    def set_density_matrix(self, rho):
        r'''Set the density matrix

        Parameters:
            rho (np.ndarray): the density matrix, shape=(dim,dim)
        '''
        assert rho.shape == (self.dim, self.dim)
        assert np.abs(rho - rho.T.conj()).max() < 1e-10
        assert abs(np.trace(rho) - 1) < 1e-10
        assert np.linalg.eigvalsh(rho)[0] > -1e-10
        EVL,EVC = np.linalg.eigh(rho)
        EVL = np.maximum(0, EVL[-self.rank:])
        assert abs(EVL.sum()-1) < 1e-10
        EVC = EVC[:,-self.rank:]
        tmp0 = (EVC * np.sqrt(EVL)).reshape(self.dim, self.rank)
        self._sqrt_rho_Tconj = torch.tensor(np.ascontiguousarray(tmp0.T.conj()), dtype=self.cdtype)

    def forward(self):
        mat_st = self.manifold()
        tmp0 = mat_st @ self._sqrt_rho_Tconj
        p_alpha_i = tmp0.real**2 + tmp0.imag**2
        p_alpha = p_alpha_i.sum(axis=1)
        tmp0 = torch.dot(p_alpha, torch.log(torch.maximum(p_alpha, self._eps)))
        tmp1 = p_alpha_i.reshape(-1)
        ret = tmp0 - torch.dot(tmp1, torch.log(torch.maximum(tmp1, self._eps)))
        return ret


def get_geometric_coherence_pure(psi:np.ndarray) -> float:
    assert psi.ndim==1
    tmp0 = psi.real**2 + psi.imag**2
    ret = 1 - tmp0.max()
    return ret


def get_geometric_coherence_1qubit(rho:np.ndarray) -> float:
    assert rho.shape == (2,2)
    ret = 0.5 - 0.5*np.sqrt(max(0, 1-4*abs(rho[0,1])**2))
    return ret


class GeometricCoherenceModel(torch.nn.Module):
    def __init__(self, dim:int, num_term:int, temperature:float|None=None, rank:int|None=None):
        super().__init__()
        self.dtype = torch.float64
        self.cdtype = torch.complex128
        self.dim = int(dim)
        self.num_term = int(num_term)
        if rank is None:
            rank = self.dim
        assert num_term>=rank
        assert rank!=1, 'for pure state, call "numqi.entangle.get_geometric_coherence_pure()" instead'
        self.manifold = numqi.manifold.Stiefel(num_term, rank, dtype=self.cdtype, method='polar')
        self.rank = rank
        self.temperature = temperature

        self._sqrt_rho_Tconj = None
        self._eps = torch.tensor(torch.finfo(self.dtype).smallest_normal, dtype=self.dtype)

    def set_density_matrix(self, rho):
        r'''Set the density matrix

        Parameters:
            rho (np.ndarray): the density matrix, shape=(dim,dim)
        '''
        assert rho.shape == (self.dim, self.dim)
        assert np.abs(rho - rho.T.conj()).max() < 1e-10
        assert abs(np.trace(rho) - 1) < 1e-10
        assert np.linalg.eigvalsh(rho)[0] > -1e-10
        EVL,EVC = np.linalg.eigh(rho)
        EVL = np.maximum(0, EVL[-self.rank:])
        assert abs(EVL.sum()-1) < 1e-10
        EVC = EVC[:,-self.rank:]
        tmp0 = (EVC * np.sqrt(EVL)).reshape(self.dim, self.rank)
        self._sqrt_rho_Tconj = torch.tensor(np.ascontiguousarray(tmp0.T.conj()), dtype=self.cdtype)

    def forward(self, use_temperature=True):
        mat_st = self.manifold()
        tmp0 = mat_st @ self._sqrt_rho_Tconj
        p_alpha_i = tmp0.real**2 + tmp0.imag**2
        if use_temperature and (self.temperature is not None):
            ret = 1 - torch.logsumexp(p_alpha_i/self.temperature, dim=1).sum()
        else:
            ret = 1 - p_alpha_i.max(axis=1)[0].sum()
        return ret


def get_real_equal_prob_state_geometric_coherence(dim:int, alpha:float):
    tmp0 = (dim-1) * np.sqrt(np.maximum(1-alpha, 0))
    tmp1 = np.sqrt(np.maximum(1 + (dim-1)*alpha, 0))
    ret = 1 - (tmp0 + tmp1)**2 / (dim*dim)
    return ret
