import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

import numqi

from utils import get_hyperdeterminant

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
        tmp0 = torch.finfo(self.dtype).smallest_normal**(1/2) #1/2 is chosen arbitrarily
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


def _get_3tangle_GHZ_W_hf0(abcdf, p):
    a, b, c, d, f = abcdf
    tmp0 = 1/np.sqrt(abs(a)**2+abs(b)**2)
    a,b = a*tmp0, b*tmp0
    tmp0 = 1/np.sqrt(abs(c)**2 + abs(d)**2 + abs(f)**2)
    c,d,f = c*tmp0, d*tmp0, f*tmp0
    assert 0<=p<=1
    return a,b,c,d,f,p
    pass

def get_3tangle_GHZ_W_pure(abcdf, p, phi):
    # https://doi.org/10.1088/1367-2630/10/4/043014
    a,b,c,d,f,p = _get_3tangle_GHZ_W_hf0(abcdf, p)
    tmp0 = np.sqrt(np.maximum(p*((1-p)**3), 0))
    ret = 4 * np.abs(p*p*a*a*b*b - 4*tmp0*np.exp(3j*phi)*b*c*d*f)
    return ret

def get_GHZ_W_state_pure(abcdf, p, phi):
    # https://doi.org/10.1088/1367-2630/10/4/043014
    a,b,c,d,f,p = _get_3tangle_GHZ_W_hf0(abcdf, p)
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
    a,b,c,d,f,p = _get_3tangle_GHZ_W_hf0(abcdf, p)
    ghz = np.zeros(8, dtype=np.complex128)
    ghz[0] = a
    ghz[7] = b
    psiw = np.zeros(8, dtype=np.complex128)
    psiw[1] = c
    psiw[2] = d
    psiw[4] = f
    ret = p*ghz.reshape(8,1) * ghz.conj() + (1-p)*psiw.reshape(8,1) * psiw.conj()
    return ret


def get_3tangle_GHZ_W_mixed(abcdf, p, eps=1e-10):
    # https://doi.org/10.1088/1367-2630/10/4/043014
    # TODO assert abcdf to be real
    a,b,c,d,f,p = _get_3tangle_GHZ_W_hf0(abcdf, p)
    if (abs(a)<eps) or (abs(b)<eps):
        ret = 0
    else:
        aa,ab,ac,ad,af = [abs(x) for x in [a,b,c,d,f]]
        s = 4*ac*ad*af / (aa*aa*ab)
        p0 = (16*ac*ac*ad*ad*af*af)**(1/3) / ((aa*aa*aa*aa*ab*ab)**(1/3) + (16*ac*ac*ad*ad*af*af)**(1/3))
        tmp0 = (aa*aa*aa*aa+ab*ab+16*ac*ac*ad*ad*af*af) / (aa*aa*aa*aa+ab*ab)
        p1 = max(p0, 0.5*(1+tmp0**(-1/2)))
        if p<=p0:
            ret = 0
        elif p>=p1:
            ret = get_3tangle_GHZ_W_pure(abcdf, p, 0)
        else:
            t_ghz = 4*aa*aa*ab*ab
            tmp0 = (p1*p1 - np.sqrt(p1*(1-p1)**3) * s)
            ret = t_ghz*(p-p1)/(1-p1) + tmp0*(1-p)/(1-p1)
    return ret




def test_get_3tangle_GHZ_W_pure():
    theta_list = np.linspace(0, np.pi, 10)
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
    assert np.abs(ret_analytical-ret_opt).max()<1e-10


def bug_test_get_3tangle_GHZ_W_mixed():
    # TODO bug unsolved
    s12 = 1/np.sqrt(2)
    s13 = 1/np.sqrt(3)
    abcdf = np.array([s12, s12, s13, s13, s13])
    plist = np.linspace(0, 1, 30)
    ret_analytical = np.array([get_3tangle_GHZ_W_mixed(abcdf, p) for p in plist])

    model = ThreeTangleModel(num_term=4*8)
    kwargs = dict(theta0='uniform', num_repeat=3, tol=1e-10, print_every_round=0)
    ret_opt = []
    for pi in tqdm(plist):
        model.set_density_matrix(get_GHZ_W_mixed(abcdf, pi))
        ret_opt.append(numqi.optimize.minimize(model, **kwargs).fun)
    ret_opt = np.array(ret_opt)

    fig,ax = plt.subplots()
    ax.plot(plist, ret_analytical, 'x', label='analytical')
    ax.plot(plist, ret_opt, label='manifold-opt')
    ax.legend()
    ax.set_xlabel('p')
    ax.set_ylabel('3-tangle')
    fig.tight_layout()
    fig.savefig('tbd02.png', dpi=200)


def demo_misc00():
    tmp0 = np.zeros(8, dtype=np.float64)
    tmp0[[0,7]] = 1/np.sqrt(2)
    dm_target = tmp0.reshape(8,1) * tmp0.conj()
    alpha_list = np.linspace(0, 1, 100)

    model = ThreeTangleModel(num_term=4*8)
    ret_opt = []
    for alpha_i in tqdm(alpha_list):
        model.set_density_matrix(numqi.entangle.hf_interpolate_dm(dm_target, alpha=alpha_i))
        ret_opt.append(numqi.optimize.minimize(model, 'uniform', num_repeat=3, tol=1e-10, print_every_round=0).fun)
        # numqi.optimize.minimize_adam(model, num_step=5000, theta0='uniform', optim_args=('adam', 0.01, 0.001), early_stop_threshold=1e-7)
    ret_opt = np.array(ret_opt)

    fig,ax = plt.subplots()
    ax.plot(alpha_list, ret_opt, 'x', label='manifold-opt')
    ax.legend()
    ax.set_xlabel('alpha')
    ax.set_ylabel('3-tangle')
    ax.set_title('GHZ state')
    ax.set_yscale('log')
    fig.tight_layout()
    fig.savefig('tbd02.png', dpi=200)
