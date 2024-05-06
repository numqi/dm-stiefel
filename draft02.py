import functools
import numpy as np
import torch
import opt_einsum
from tqdm import tqdm
import matplotlib.pyplot as plt

import numqi

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


def test_get_hyperdeterminant():
    N0 = 3
    np0 = np.random.randn(N0, 8) + 1j*np.random.randn(N0, 8)
    ret0 = get_hyperdeterminant(np0, mode='tensor')
    ret1 = get_hyperdeterminant(np0, mode='index')
    assert np.abs(ret0-ret1).max()<1e-10
    ret2 = get_hyperdeterminant(torch.tensor(np0, dtype=torch.complex128), mode='tensor').numpy()
    ret3 = get_hyperdeterminant(torch.tensor(np0, dtype=torch.complex128), mode='index').numpy()
    assert np.abs(ret0-ret2).max()<1e-10
    assert np.abs(ret0-ret3).max()<1e-10

    np0 = np.random.randn(N0, 8)
    ret0 = get_hyperdeterminant(np0, mode='tensor')
    ret1 = get_hyperdeterminant(np0, mode='index')
    assert np.abs(ret0-ret1).max()<1e-10
    ret2 = get_hyperdeterminant(torch.tensor(np0, dtype=torch.float64), mode='tensor').numpy()
    ret3 = get_hyperdeterminant(torch.tensor(np0, dtype=torch.float64), mode='index').numpy()
    assert np.abs(ret0-ret2).max()<1e-10
    assert np.abs(ret0-ret3).max()<1e-10


def get_3tangle_pure(np0:np.ndarray|torch.Tensor, mode:str='tensor'):
    tmp0 = get_hyperdeterminant(np0, mode)
    tmp1 = tmp0.real**2 + tmp0.imag**2
    if isinstance(tmp1, torch.Tensor):
        ret = 4*torch.sqrt(tmp1)
    else:
        ret = 4*np.sqrt(tmp1)
    return ret

# class


def test_concurrence_2qubit_pure():
    # http://dx.doi.org/10.1103/PhysRevA.86.042302
    psi = numqi.random.rand_haar_state(4)
    z0 = numqi.entangle.get_concurrence_pure(psi.reshape(2,2))
    assert abs(np.vdot(psi, np.kron(numqi.gate.Y, numqi.gate.Y) @ psi.conj())) < 1e-10

psi = numqi.random.rand_haar_state(8)


class ThreeTangleModel(torch.nn.Module):
    def __init__(self, num_term:int, rank:int|None=None):
        super().__init__()
        self.dtype = torch.float64
        self.cdtype = torch.complex128
        self.num_term = int(num_term)
        if rank is None:
            rank = 8
        assert num_term>=rank
        assert rank!=1, 'for pure state, call "numqi.entangle.get_3tangle_pure()" instead'
        self.manifold = numqi.manifold.Stiefel(num_term, rank, dtype=self.cdtype, method='polar')
        self.rank = rank

        self._sqrt_rho_Tconj = None
        tmp0 = torch.finfo(self.dtype).smallest_normal
        self._eps0 = torch.tensor(tmp0, dtype=self.dtype) #in this way sqrt(_eps0)/_eps1 is a small number
        self._eps1 = torch.tensor(tmp0**(1/3), dtype=self.dtype)

    def set_density_matrix(self, rho):
        r'''Set the density matrix

        Parameters:
            rho (np.ndarray): the density matrix, shape=(dim,dim)
        '''
        assert rho.shape == (8,8)
        assert np.abs(rho - rho.T.conj()).max() < 1e-10
        assert abs(np.trace(rho) - 1) < 1e-10
        assert np.linalg.eigvalsh(rho)[0] > -1e-10
        EVL,EVC = np.linalg.eigh(rho)
        EVL = np.maximum(0, EVL[-self.rank:])
        assert abs(EVL.sum()-1) < 1e-10
        EVC = EVC[:,-self.rank:]
        tmp0 = (EVC * np.sqrt(EVL)).reshape(8, self.rank)
        self._sqrt_rho_Tconj = torch.tensor(np.ascontiguousarray(tmp0.T.conj()), dtype=self.cdtype)

    def forward(self):
        mat_st = self.manifold()
        psi_tilde = mat_st @ self._sqrt_rho_Tconj
        tmp0 = get_hyperdeterminant(psi_tilde, mode='tensor')
        tmp1 = tmp0.real**2 + tmp0.imag**2
        tau_tilde = 4*torch.sqrt(torch.maximum(tmp1, self._eps0))
        tmp2 = torch.maximum((psi_tilde.real**2 + psi_tilde.imag**2).sum(axis=1), self._eps1)
        ret = (tau_tilde / tmp2).sum()
        return ret

def get_3tangle_GHZ_W_pure(abcdf, p, phi):
    # https://doi.org/10.1088/1367-2630/10/4/043014
    a, b, c, d, f = abcdf
    tmp0 = 1/np.sqrt(a*a+b*b)
    a,b = a*tmp0, b*tmp0
    tmp0 = 1/np.sqrt(c*c + d*d + f*f)
    c,d,f = c*tmp0, d*tmp0, f*tmp0
    tmp0 = np.sqrt(np.maximum(p*((1-p)**3), 0))
    ret = 4 * np.abs(p*p*a*a*b*b - 4*tmp0*np.exp(3j*phi)*b*c*d*f)
    return ret

def get_GHZ_W_state_pure(abcdf, p, phi):
    # https://doi.org/10.1088/1367-2630/10/4/043014
    # TODO abcdf could be complex
    a, b, c, d, f = abcdf
    tmp0 = 1/np.sqrt(a*a+b*b)
    a,b = a*tmp0, b*tmp0
    tmp0 = 1/np.sqrt(c*c + d*d + f*f)
    c,d,f = c*tmp0, d*tmp0, f*tmp0
    assert 0<=p<=1
    ret = np.zeros(8, dtype=np.complex128)
    ret[0] = a*np.sqrt(p)
    ret[7] = b*np.sqrt(p)
    tmp0 = -np.sqrt(max(0, 1-p)) * np.exp(1j*phi)
    ret[1] = c*tmp0
    ret[2] = d*tmp0
    ret[4] = f*tmp0
    return ret


def get_GHZ_W_mixed(abcdf, p):
    # https://doi.org/10.1088/1367-2630/10/4/043014
    pass


def get_3tangle_GHZ_W_mixed():
    # https://doi.org/10.1088/1367-2630/10/4/043014
    pass




#def test_get_3tangle_GHZ_W_pure():

theta_list = np.linspace(0, np.pi, 50)
model = ThreeTangleModel(num_term=4*8)

ret_analytical = []
ret_opt = []
kwargs = dict(theta0='uniform', num_repeat=3, tol=1e-10, print_every_round=0)
for theta_i in theta_list:
    abcdf = np.cos(theta_i), np.sin(theta_i), 1, 0, 0
    p = 1
    phi = 0
    ret_analytical.append(get_3tangle_GHZ_W_pure(abcdf, p, phi))
    psi = get_GHZ_W_state_pure(abcdf, p, phi)
    model.set_density_matrix(psi.reshape(8,1) * psi.conj())
    ret_opt.append(numqi.optimize.minimize(model, **kwargs).fun)
ret_analytical = np.array(ret_analytical)
ret_opt = np.array(ret_opt)

fig,ax = plt.subplots()
ax.plot(theta_list, ret_opt, label='manifold-opt')
ax.plot(theta_list, ret_analytical, 'x', label='analytical')
ax.legend()
ax.set_xlabel('theta')
ax.set_ylabel('3-tangle')
ax.set_title('GHZ/W state')
fig.tight_layout()
fig.savefig('tbd02.png', dpi=200)
