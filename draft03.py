import numpy as np
import torch
import scipy.sparse
import opt_einsum
from tqdm import tqdm
import matplotlib.pyplot as plt

import numqi


def scipy_sparse_csr_to_torch(np0, dtype):
    assert (np0.ndim==2) and (np0.format=='csr')
    tmp0 = torch.tensor(np0.indptr, dtype=torch.int64)
    tmp1 = torch.tensor(np0.indices, dtype=torch.int64)
    tmp2 = torch.tensor(np0.data, dtype=dtype)
    ret = torch.sparse_csr_tensor(tmp0, tmp1, tmp2, dtype=dtype)
    return ret


def test_MagicStabilizerEntropyModel():
    num_qubit = 2
    rho = numqi.random.rand_density_matrix(2**num_qubit)
    alpha = 2
    model = MagicStabilizerEntropyModel(alpha, num_qubit, num_term=4*(2**num_qubit))
    model.set_density_matrix(rho)
    ret0 = model().item()
    mat_st = model.manifold().detach()
    psi_tilde = (mat_st @ model._sqrt_rho_Tconj).numpy().copy().conj()
    plist = np.linalg.norm(psi_tilde, axis=1, ord=2)**2
    assert abs(plist.sum()-1) < 1e-10
    psi_list = psi_tilde / np.sqrt(plist[:,None])
    pauli_mat = numqi.gate.get_pauli_group(num_qubit)
    z0 = np.einsum(psi_list.conj(), [0,1], pauli_mat, [3,1,2], psi_list, [0,2], [0,3], optimize=True)
    assert np.abs(z0.imag).max() < 1e-12
    ret_ = np.dot(plist, ((z0.real**2)**alpha).sum(axis=1)) / (2**num_qubit)
    assert abs(ret0+ret_) < 1e-10


def hf_torch_real_soft_maximum(x, epsilon, temperature:float):
    assert (x.ndim==1) and (epsilon.ndim==0)
    if temperature is None:
        ret = torch.maximum(x, epsilon)
    else:
        tmp0 = torch.stack([torch.ones_like(x)*epsilon, x], axis=1)
        ret = temperature*torch.logsumexp(tmp0 / temperature, dim=1)
    return ret


class MagicStabilizerEntropyModel(torch.nn.Module):
    def __init__(self, alpha:float, num_qubit:int, num_term:int, rank:int|None=None, method:str='polar'):
        super().__init__()
        self.dtype = torch.float64
        self.cdtype = torch.complex128
        assert alpha>=2
        self.alpha = float(alpha)
        self.num_qubit = int(num_qubit)
        self.num_term = int(num_term)
        if rank is None:
            rank = 2**self.num_qubit
        assert num_term>=rank
        self.manifold = numqi.manifold.Stiefel(num_term, rank, dtype=self.cdtype, method=method)
        self.rank = rank

        tmp0 = numqi.gate.get_pauli_group(num_qubit, use_sparse=True)
        self.pauli_mat = scipy.sparse.vstack(tmp0, format='csr')

        self._sqrt_rho_Tconj = None
        self._psi_pauli_psi = None
        self.contract_expr = None
        # 1/6 is arbitrarily chosen
        # self._eps = torch.tensor(torch.finfo(self.dtype).smallest_normal**(1/6), dtype=self.dtype)
        self._eps = torch.tensor(10**(-12), dtype=self.dtype)
        self.temperature = 0.1

    def set_density_matrix(self, rho):
        r'''Set the density matrix

        Parameters:
            rho (np.ndarray): the density matrix, shape=(dim,dim)
        '''
        dim = 2**self.num_qubit
        assert rho.shape == (dim, dim)
        assert np.abs(rho - rho.T.conj()).max() < 1e-10
        assert abs(np.trace(rho) - 1) < 1e-10
        assert np.linalg.eigvalsh(rho)[0] > -1e-10
        EVL,EVC = np.linalg.eigh(rho)
        EVL = np.maximum(0, EVL[-self.rank:])
        assert abs(EVL.sum()-1) < 1e-10
        EVC = EVC[:,-self.rank:]
        tmp0 = (EVC * np.sqrt(EVL)).reshape(dim, self.rank)
        self._sqrt_rho_Tconj = torch.tensor(np.ascontiguousarray(tmp0.T.conj()), dtype=self.cdtype)
        tmp1 = (self.pauli_mat @ tmp0).reshape(-1, dim, self.rank)
        self._psi_pauli_psi = torch.tensor(np.einsum(tmp1, [0,1,2], tmp0.conj(), [1,3], [0,3,2], optimize=True), dtype=self.cdtype)
        # man_st, mat_st.conj(), _psi_pauli_psi
        self.contract_expr = opt_einsum.contract_expression([self.num_term,self.rank], [0,1],
                    [self.num_term,self.rank], [0,2], self._psi_pauli_psi, [3,1,2], [3,0], constants=[2])

    def forward(self, use_temperature=True):
        mat_st = self.manifold()
        tmp0 = ((self.contract_expr(mat_st, mat_st.conj()).real**2)**self.alpha).sum(axis=0)
        psi_tilde = mat_st @ self._sqrt_rho_Tconj
        plist = (psi_tilde.real**2 + psi_tilde.imag**2).sum(axis=1)
        # tmp2 = hf_torch_real_soft_maximum(plist, self._eps, self.temperature if use_temperature else None)**(1-2*self.alpha)
        tmp2 = torch.maximum(plist, self._eps)**(1-2*self.alpha)
        # tmp2 = plist**(1-2*self.alpha)
        loss = - torch.dot(tmp0, tmp2) / (2**self.num_qubit)
        return loss


def demo_H_state():
    alpha_list = [2,3,4]
    # alpha = 2
    Hstate = np.array([1, np.sqrt(2)-1]) / np.sqrt(4-2*np.sqrt(2))
    dm_target = Hstate.reshape(-1,1) * Hstate
    # dm_target = np.eye(2)/2 + np.array([[1,1], [1,-1]]) / (2*np.sqrt(2))
    num_qubit = numqi.utils.hf_num_state_to_num_qubit(dm_target.shape[0])
    prob_list = np.linspace(0, 1, 100)

    ret_opt = []
    for alpha_i in alpha_list:
        model = MagicStabilizerEntropyModel(alpha_i, num_qubit, num_term=4*(2**num_qubit))
        for prob_i in tqdm(prob_list):
            model.set_density_matrix(numqi.entangle.hf_interpolate_dm(dm_target, alpha=prob_i))
            ret_opt.append(-numqi.optimize.minimize(model, 'uniform', num_repeat=3, tol=1e-10, print_every_round=0).fun)
    ret_opt = np.array(ret_opt).reshape(len(alpha_list), -1)

    fig,ax = plt.subplots()
    ax.axvline(1/np.sqrt(2), color='red', label='1/sqrt(2)')
    for ind0 in range(len(alpha_list)):
        ax.plot(prob_list, 1-ret_opt[ind0], label=f'alpha={alpha_list[ind0]}')
    ax.set_xlabel(r'$p\rho + (1-p)I/d$')
    ax.set_ylabel('linear Stab Entropy')
    ax.set_title(f'H state')
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
        model = MagicStabilizerEntropyModel(alpha_i, num_qubit, num_term=4*(2**num_qubit), method='polar')
        for prob_i in tqdm(prob_list):
            model.set_density_matrix(numqi.entangle.hf_interpolate_dm(dm_target, alpha=prob_i))
            ret_opt.append(-numqi.optimize.minimize(model, 'uniform', num_repeat=10, tol=1e-10, print_every_round=0).fun)
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
        model = MagicStabilizerEntropyModel(alpha_i, num_qubit, num_term=4*(2**num_qubit), method='so-exp')
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
