import functools
import numpy as np
import torch
import opt_einsum
import scipy.sparse
import cvxpy

import numqi
import cvxpy as cp

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
            ret = 1 - self.temperature*torch.logsumexp(p_alpha_i/self.temperature, dim=1).sum()
        else:
            ret = 1 - p_alpha_i.max(axis=1)[0].sum()
        return ret


def get_maximally_coherent_state(dim:int):
    r'''Get the maximally coherent state

    reference: Maximally coherent states and coherence-preserving operations
    [arxiv-link](https://arxiv.org/abs/1511.02576)

    Parameters:
        dim (int): the dimension of the state

    Returns:
        ret (np.ndarray): the maximally coherent state, shape=(dim,)
    '''
    assert dim>=1
    ret = np.ones(dim, dtype=np.float64)/np.sqrt(dim)
    return ret


def get_maximally_coherent_state_mixed(dim:int, alpha:float):
    r'''Get the interpolation between the maximally coherent state and the maximally mixed state

    reference: Numerical and analytical results for geometric measure of coherence and geometric measure of entanglement
    [arxiv-link](https://arxiv.org/abs/1903.10944)

    Parameters:
        dim (int): the dimension of the state
        alpha (float): the interpolation parameter

    Returns:
        ret (np.ndarray): density matrix, shape=(dim,dim)
    '''
    assert dim>=1
    rho = np.ones((dim,dim), dtype=np.float64)/dim
    ret = numqi.entangle.hf_interpolate_dm(rho, alpha=alpha)
    return ret

def get_maximally_coherent_state_mixed_coherence(dim:int, alpha:float|np.ndarray):
    r'''Get the geometric measure of coherence for the maximally coherent state (mcs) along the interpolation with maximally mixed state

    reference: Numerical and analytical results for geometric measure of coherence and geometric measure of entanglement
    [arxiv-link](https://arxiv.org/abs/1903.10944)

    Parameters:
        dim (int): the dimension of the state
        alpha (float,np.ndarray): the interpolation parameter

    Returns:
        ret (float,np.ndarray): the coherence
    '''
    tmp0 = (dim-1) * np.sqrt(np.maximum(1-alpha, 0))
    tmp1 = np.sqrt(np.maximum(1 + (dim-1)*alpha, 0))
    ret = 1 - (tmp0 + tmp1)**2 / (dim*dim)
    return ret


def _get_hyperdeterminant_index_print():
    hf0 = lambda x: [int(y, base=2) for y in x.split(' ')]
    tmp0 = '''
    1   1   1   1   -2  -2  -2  -2  -2  -2  4   4
    000 001 010 100 000 000 000 011 011 101 000 111
    000 001 010 100 111 111 111 100 100 010 110 001
    111 110 101 011 011 101 110 101 110 110 101 010
    111 110 101 011 100 010 001 010 001 001 011 100
    '''
    tmp1 = tmp0.strip().split('\n')
    coeff = np.array([int(x) for x in tmp1[0].split()])
    index = np.array([hf0(x) for x in tmp1[1:]])
    print(coeff)
    print(index)


@functools.lru_cache
def _get_hyperdeterminant_index_cache(dtype):
    # see _get_hyperdeterminant_index_print() for details
    tmp0 = {np.float32, np.float64, np.complex64, np.complex128}
    tmp1 = {torch.float32, torch.float64, torch.complex64, torch.complex128}
    assert (dtype in tmp0) or (dtype in tmp1)
    coeff = np.array([ 1,  1,  1,  1, -2, -2, -2, -2, -2, -2,  4,  4], dtype=np.float64)
    index = np.array([[0, 1, 2, 4, 0, 0, 0, 3, 3, 5, 0, 7],
        [0, 1, 2, 4, 7, 7, 7, 4, 4, 2, 6, 1],
        [7, 6, 5, 3, 3, 5, 6, 5, 6, 6, 5, 2],
        [7, 6, 5, 3, 4, 2, 1, 2, 1, 1, 3, 4]], dtype=np.int64)
    if dtype in tmp1:
        coeff = torch.tensor(coeff, dtype=dtype)
        index = torch.tensor(index, dtype=torch.int64)
    return coeff, index


def _get_hyperdeterminant_index(np0):
    assert np0.shape[-1]==8
    shape = np0.shape
    np0 = np0.reshape(-1, 8)
    tmp0 = np0.dtype.type if isinstance(np0, np.ndarray) else np0.dtype
    coeff, index = _get_hyperdeterminant_index_cache(tmp0)
    ret = (np0[:,index[0]] * np0[:,index[1]] * np0[:,index[2]] * np0[:,index[3]]) @ coeff
    ret = ret[0] if (len(shape)==1) else ret.reshape(shape[:-1])
    return ret


@functools.lru_cache
def _get_hyperdeterminant_tensor_cache(N0:int, dtype):
    tmp0 = {np.float32, np.float64, np.complex64, np.complex128}
    tmp1 = {torch.float32, torch.float64, torch.complex64, torch.complex128}
    assert (dtype in tmp0) or (dtype in tmp1)
    if dtype in tmp1: #is_torch=True
        epsilon = torch.tensor([[0,1],[-1,0]], dtype=dtype)
    else:
        epsilon = np.array([[0,1],[-1,0]], dtype=dtype)
    expr0 = opt_einsum.contract_expression([N0,2,2,2], [0,1,2,3], [N0,2,2,2], [0,4,5,6], epsilon, [1,4], epsilon, [2,5], [0,3,6], constants=[2,3])
    expr1 = opt_einsum.contract_expression([N0,2,2], [0,1,2], [N0,2,2], [0,3,4], epsilon, [1,3], -epsilon*0.5, [2,4], [0], constants=[2,3])
    # weird factor -0.5
    return expr0, expr1

def _get_hyperdeterminant_tensor(np0):
    assert np0.shape[-1]==8
    shape = np0.shape
    np0 = np0.reshape(-1, 2, 2, 2)
    tmp0 = np0.dtype.type if isinstance(np0, np.ndarray) else np0.dtype
    expr0,expr1 = _get_hyperdeterminant_tensor_cache(np0.shape[0], tmp0)
    tmp0 = expr0(np0, np0)
    ret = expr1(tmp0, tmp0)
    ret = ret[0] if (len(shape)==1) else ret.reshape(shape[:-1])
    return ret


def get_hyperdeterminant(np0:np.ndarray|torch.Tensor, mode:str='tensor'):
    # https://en.wikipedia.org/wiki/Hyperdeterminant
    assert mode in {'tensor', 'index'}
    if mode=='tensor':
        ret = _get_hyperdeterminant_tensor(np0)
    elif mode=='index':
        ret = _get_hyperdeterminant_index(np0)
    return ret


def get_3tangle_pure(np0:np.ndarray|torch.Tensor, mode:str='tensor'):
    tmp0 = get_hyperdeterminant(np0, mode)
    tmp1 = tmp0.real**2 + tmp0.imag**2
    if isinstance(tmp1, torch.Tensor):
        ret = 4*torch.sqrt(tmp1)
    else:
        ret = 4*np.sqrt(tmp1)
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
        self._eps = torch.tensor(torch.finfo(self.dtype).smallest_normal**(1/6), dtype=self.dtype)
        # self._eps = torch.tensor(10**(-12), dtype=self.dtype)

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

    def forward(self):
        mat_st = self.manifold()
        tmp0 = ((self.contract_expr(mat_st, mat_st.conj()).real**2)**self.alpha).sum(axis=0)
        psi_tilde = mat_st @ self._sqrt_rho_Tconj
        plist = (psi_tilde.real**2 + psi_tilde.imag**2).sum(axis=1)
        tmp2 = torch.maximum(plist, self._eps)**(1-2*self.alpha)
        loss = - torch.dot(tmp0, tmp2) / (2**self.num_qubit)
        return loss


def scipy_sparse_csr_to_torch(np0, dtype):
    assert (np0.ndim==2) and (np0.format=='csr')
    tmp0 = torch.tensor(np0.indptr, dtype=torch.int64)
    tmp1 = torch.tensor(np0.indices, dtype=torch.int64)
    tmp2 = torch.tensor(np0.data, dtype=dtype)
    ret = torch.sparse_csr_tensor(tmp0, tmp1, tmp2, dtype=dtype)
    return ret


def get_geometric_measure_coherence_sdp(rho:np.ndarray):
    r'''Get the geometric measure of coherence using semi-definite programming (SDP)

    reference: Numerical and analytical results for geometric measure of coherence and geometric measure of entanglement
    [arxiv-link](https://arxiv.org/abs/1903.10944)

    Parameters:
        rho (np.ndarray): the density matrix, shape=(dim,dim), support batch input (batch,dim,dim)

    Returns:
        ret (float,np.ndarray): the coherence
    '''
    assert rho.ndim in {2,3}
    isone = (rho.ndim == 2)
    if isone:
        rho = rho[None]
    assert np.abs(rho-rho.transpose(0,2,1).conj()).max() < 1e-10
    dim = rho.shape[-1]
    cvxD = cvxpy.Variable((dim,dim), diag=True)
    cvxX = cvxpy.Variable((dim,dim), complex=True)
    cvxrho = cvxpy.Parameter((dim,dim), PSD=True) #hermitian=True
    constraint = [
        cvxpy.bmat([[cvxrho, cvxX], [cvxpy.conj(cvxX.T), cvxD]]) >> 0,
        cvxD >> 0,
        cvxpy.trace(cvxD) == 1
    ]
    obj = cvxpy.Maximize(cvxpy.real(cvxpy.trace(cvxX)))
    prob = cvxpy.Problem(obj, constraint)
    ret = []
    for x in rho:
        cvxrho.value = x
        prob.solve()
        ret.append(1 - obj.value**2)
    ret = ret[0] if isone else np.array(ret)
    return ret
