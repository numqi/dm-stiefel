import time
import numpy as np
import torch
import cvxpy
from tqdm import tqdm
import matplotlib.pyplot as plt

import numqi
from zzz233 import to_pickle, from_pickle

np_rng = np.random.default_rng()
cp_tableau = ['#006BA4', '#FF800E', '#ABABAB', '#595959', '#5F9ED1', '#C85200', '#898989', '#A2C8EC', '#FFBC79', '#CFCFCF']

# linear entropy http://dx.doi.org/10.1103/PhysRevLett.114.160501


def demo_werner_convex_concave():
    alpha_list = np.linspace(0, 1, 100)
    dim = 3

    model = numqi.entangle.DensityMatrixLinearEntropyModel([dim,dim], num_ensemble=27, kind='convex')
    ret0 = []
    for alpha_i in tqdm(alpha_list):
        model.set_density_matrix(numqi.state.Werner(dim, alpha=alpha_i))
        ret0.append(numqi.optimize.minimize(model, num_repeat=3, tol=1e-10, print_every_round=0).fun)
    ret0 = np.array(ret0)

    model = numqi.entangle.DensityMatrixLinearEntropyModel([dim,dim], num_ensemble=27, kind='concave')
    ret1 = []
    for alpha_i in tqdm(alpha_list):
        model.set_density_matrix(numqi.state.Werner(dim, alpha=alpha_i))
        ret1.append(-numqi.optimize.minimize(model, num_repeat=3, tol=1e-10, print_every_round=0).fun)
    ret1 = np.array(ret1)


    fig,ax = plt.subplots()
    ax.axvline(1/dim, color='r')
    ax.plot(alpha_list, ret0, label='convex')
    ax.plot(alpha_list, ret1, label='concave')
    ax.legend()
    # ax.set_yscale('log')
    ax.set_xlabel('alpha')
    ax.set_ylabel('linear entropy')
    ax.set_title(f'Werner({dim})')
    fig.tight_layout()
    fig.savefig('tbd00.png', dpi=200)


def demo_Horodecki1997_3x3_contourf():
    alist = np.linspace(0, 1, 30)
    plist = np.linspace(0.92, 1, 30)

    ret = []
    model = numqi.entangle.DensityMatrixLinearEntropyModel([3,3], num_ensemble=27, kind='convex')
    tmp0 = [(a,p) for a in alist for p in plist]
    for a,p in tqdm(tmp0):
        rho = numqi.state.get_bes3x3_Horodecki1997(a)
        model.set_density_matrix(numqi.entangle.hf_interpolate_dm(rho, alpha=p))
        ret.append(numqi.optimize.minimize(model, num_repeat=3, tol=1e-10, print_every_round=0).fun)
    ret = np.array(ret).reshape(len(alist), len(plist))


    fig,ax = plt.subplots()
    tmp0 = np.log(np.maximum(1e-7, ret))
    hcontourf = ax.contourf(alist, plist, tmp0.T, levels=10)
    cbar = fig.colorbar(hcontourf)
    ax.set_xlabel('a')
    ax.set_ylabel('p')
    ax.set_title('manifold-opt')
    fig.tight_layout()
    fig.savefig('tbd00.png', dpi=200)


def demo_werner3():
    alpha_list = np.linspace(0, 1, 100)
    dim = 3

    tmp0 = np.stack([numqi.state.Werner(dim, alpha=alpha_i) for alpha_i in alpha_list])
    ret = numqi.entangle.get_linear_entropy_entanglement_ppt(tmp0, (dim,dim), use_tqdm=True)

    model = numqi.entangle.DensityMatrixLinearEntropyModel([dim,dim], num_ensemble=27, kind='convex', method='polar')
    ret0 = []
    for alpha_i in tqdm(alpha_list):
        model.set_density_matrix(numqi.state.Werner(dim, alpha=alpha_i))
        ret0.append(numqi.optimize.minimize(model, num_repeat=3, tol=1e-10, print_every_round=0).fun)
    ret0 = np.array(ret0)

    fig,ax = plt.subplots()
    ax.plot(alpha_list, ret, label='PPT')
    ax.plot(alpha_list, ret0, 'x', label='manifold-opt')
    ax.legend()
    ax.set_xlabel('p')
    ax.set_ylabel('linear entropy')
    ax.set_yscale('log')
    ax.set_title('Werner(3)')
    fig.tight_layout()
    fig.savefig('tbd00.png', dpi=200)


def demo_werner_compare_polar_euler():
    alpha_list = np.linspace(0, 1, 100)
    dim = 3

    model = numqi.entangle.DensityMatrixLinearEntropyModel([dim,dim], num_ensemble=27, kind='convex', method='polar')
    ret0 = []
    for alpha_i in tqdm(alpha_list):
        model.set_density_matrix(numqi.state.Werner(dim, alpha=alpha_i))
        ret0.append(numqi.optimize.minimize(model, num_repeat=3, tol=1e-10, print_every_round=0).fun)
    ret0 = np.array(ret0)

    # too slow
    model = numqi.entangle.DensityMatrixLinearEntropyModel([dim,dim], num_ensemble=27, kind='convex', method='euler')
    ret1 = []
    for alpha_i in tqdm(alpha_list):
        model.set_density_matrix(numqi.state.Werner(dim, alpha=alpha_i))
        ret1.append(numqi.optimize.minimize(model, ('uniform',-np.pi,np.pi), num_repeat=3, tol=1e-10, print_every_round=0).fun)
    ret1 = np.array(ret1)

    fig,ax = plt.subplots()
    ax.plot(alpha_list, ret0, label='polar')
    ax.plot(alpha_list, ret1, 'x', label='euler')
    ax.legend()
    ax.set_xlabel('p')
    ax.set_ylabel('linear entropy')
    ax.set_yscale('log')
    ax.set_title('Werner(3)')
    fig.tight_layout()
    fig.savefig('tbd00.png', dpi=200)


def demo_Horodecki1997_3x3():
    # rho = numqi.entangle.load_upb('tiles', return_bes=True)[1]
    rho = numqi.state.get_bes3x3_Horodecki1997(0.23)
    plist = np.linspace(0.92, 1, 30)

    tmp0 = np.stack([numqi.entangle.hf_interpolate_dm(rho,alpha=p) for p in plist])
    ret = numqi.entangle.get_linear_entropy_entanglement_ppt(tmp0, (3,3), use_tqdm=True)
    # 0.0017232268486448987

    ret0 = []
    model = numqi.entangle.DensityMatrixLinearEntropyModel([3,3], num_ensemble=27, kind='convex')
    for p in tqdm(plist):
        model.set_density_matrix(numqi.entangle.hf_interpolate_dm(rho, alpha=p))
        ret0.append(numqi.optimize.minimize(model, num_repeat=3, tol=1e-10, print_every_round=0).fun)
    ret0 = np.array(ret0)

    fig,ax = plt.subplots()
    ax.plot(plist, ret, label='manifold-opt')
    ax.plot(plist, ret0, 'x', label='PPT')
    ax.legend()
    ax.set_xlabel('p')
    ax.set_ylabel('linear entropy')
    ax.set_yscale('log')
    ax.set_title('Horodecki1997-2qutrit(0.23)')
    fig.tight_layout()
    fig.savefig('tbd00.png', dpi=200)


class BESNumEigenModel(torch.nn.Module):
    def __init__(self, basis_bos, dim, with_ppt=True, with_ppt1=True):
        super().__init__()
        dim_boson = len(basis_bos)
        self.manifold = numqi.moanifold.Sphere(dim_boson**2-1, dtype=torch.float64)
        assert np.abs(basis_bos.imag).max() < 1e-10
        self.basis_bos = torch.tensor(basis_bos.real, dtype=torch.complex128)
        choi_op = np.einsum(basis_bos, [0,1], basis_bos, [2,3], [0,1,2,3], optimize=True)
        matA,matb = numqi.channel.choi_op_to_bloch_map(choi_op)
        self.matA = torch.tensor(matA, dtype=torch.float64)
        self.matb = torch.tensor(matb, dtype=torch.float64)

        self.dim = dim
        self.rank = 9
        self.with_ppt = with_ppt
        self.with_ppt1 = with_ppt1

    def forward(self):
        dim_boson = self.basis_bos.shape[0]
        dim = self.dim
        rho_vec_norm = self.manifold()
        rho_norm = numqi.gellmann.gellmann_basis_to_dm(rho_vec_norm)
        tmp0 = torch.linalg.eigvalsh(rho_norm)
        beta0 = 1/(1-dim_boson*tmp0[0])
        tmp1 = 1/dim_boson + beta0*(tmp0 - 1/dim_boson)
        beta1 = 1/(1-(dim*dim)*tmp1[-1])
        EVL_rho0 = torch.concat([torch.zeros(dim*dim-dim_boson),tmp1])
        tmp0 = torch.flip(EVL_rho0, (0,)) #EVL_rho0[::-1]
        EVL_rho1 = 1/(dim*dim) + beta1*(tmp0-1/(dim*dim))

        rho0 = self.basis_bos.T @ numqi.gellmann.gellmann_basis_to_dm(beta0*rho_vec_norm) @ self.basis_bos
        rho1 = numqi.gellmann.gellmann_basis_to_dm(beta1*(self.matA @ (beta0*rho_vec_norm) + self.matb))
        # loss0 = torch.linalg.eigvalsh(rho0)[:(rho0.shape[0]-self.rank)].sum()
        # loss1 = torch.linalg.eigvalsh(rho1)[:(self.rank)].sum()
        loss0 = EVL_rho0[:(rho0.shape[0]-self.rank)].sum()
        loss1 = EVL_rho1[:self.rank].sum()
        loss = (loss0 + loss1)**2
        if self.with_ppt:
            tmp1 = rho0.reshape(dim,dim,dim,dim).transpose(1,3).reshape(dim*dim,-1)
            loss2 = torch.linalg.eigvalsh(tmp1)[0]**2
            loss = loss + loss2
            # without this constraint, it will not converge to BES
            tmp1 = rho1.reshape(dim,dim,dim,dim).transpose(1,3).reshape(dim*dim,-1)
            loss3 = torch.linalg.eigvalsh(tmp1)[0]**2
            loss = loss + loss3
        return loss


def demo_fail_stupid_idea():
    # symmetric subspace cannot build UPB/BES
    dim = 6
    z0 = numqi.group.symext.get_sud_symmetric_irrep_basis(dim, 2)
    basis_boson = z0[0][0]
    basis_fermi = z0[1][0]

    model = BESNumEigenModel(basis_boson, dim)
    model.rank = 20
    theta_optim = numqi.optimize.minimize(model, num_repeat=100, tol=1e-10, print_every_round=1)
    numqi.optimize.minimize_adam(model, 1000, 'uniform')


def demo_werner_eof_large():
    dim_list = [3, 5, 10, 15]
    time_list = []
    alpha_list = np.linspace(0, 1, 20)
    eof_analytical = np.stack([numqi.state.get_Werner_eof(x, alpha_list) for x in dim_list])

    eof_opt = []
    time_list = []
    kwargs = dict(num_repeat=1, tol=1e-9, print_every_round=0, print_freq=100)
    for dim in dim_list:
        model = numqi.entangle.EntanglementFormationModel(dim, dim, num_term=2*dim*dim)
        for alpha_i in alpha_list: #about 1 min
            t0 = time.time()
            model.set_density_matrix(numqi.state.Werner(dim, alpha_i))
            theta_optim = numqi.optimize.minimize(model, **kwargs)
            time_list.append(time.time()-t0)
            tmp0 = numqi.state.get_Werner_eof(dim, alpha_i)
            print(f'[dim={dim}, alpha={alpha_i:.2f}][{time_list[-1]:.1g}] {tmp0:.7f} {theta_optim.fun:.7f} {(theta_optim.fun-tmp0):.4g}')
            eof_opt.append(theta_optim.fun)
    eof_opt = np.array(eof_opt).reshape(len(dim_list), -1)
    time_list = np.array(time_list).reshape(len(dim_list), -1)

    fig,ax = plt.subplots()
    for ind0 in range(len(dim_list)):
        tmp0 = f'{dim_list[ind0]}x{dim_list[ind0]} (avg={time_list[ind0].mean():.2f}s)'
        ax.plot(alpha_list, eof_analytical[ind0], 'x', color=cp_tableau[ind0])
        ax.plot(alpha_list, eof_opt[ind0], label=tmp0, color=cp_tableau[ind0])
    ax.legend()
    ax.set_xlabel(r'$\alpha$')
    ax.set_yscale('log')
    ax.set_ylabel('EOF')
    ax.set_title(f'Werner, "x" for analytical, "-" for manifold-opt')
    fig.tight_layout()
    fig.savefig('data/werner_eof_large.png', dpi=200)


def demo_fail00():
    dim = 4
    tmp0 = numqi.group.symext.get_sud_symmetric_irrep_basis(dim, 2)
    basis_boson = tmp0[0][0]
    basis_fermion = tmp0[1][0]
    project_fermion = basis_fermion.T @ basis_fermion #basis are real

    cvx_rho_fermion = cvxpy.Parameter((dim*dim,dim*dim), hermitian=True)
    cvx_sigma = cvxpy.Variable((dim*dim,dim*dim), hermitian=True)
    tmp0 = cvxpy.sum(cvxpy.real(cvxpy.multiply(project_fermion, cvx_sigma)))
    obj = cvxpy.Maximize(tmp0)
    constraint = [
        cvxpy.trace(cvx_sigma) == 1,
        cvx_sigma >> 0,
        cvxpy.partial_transpose(cvx_sigma, [dim,dim], 1) >> 0,
        project_fermion @ cvx_sigma @ project_fermion == tmp0*cvx_rho_fermion,
    ]
    prob = cvxpy.Problem(obj, constraint)

    for _ in range(100):
        tmp0 = numqi.random.rand_density_matrix(len(basis_fermion))
        rho_fermion = basis_fermion.T @ tmp0 @ basis_fermion
        cvx_rho_fermion.value = rho_fermion

        optim_value = prob.solve()
        print(optim_value)
        if optim_value < (0.5-1e-7):
            break

    zero_eps = 1e-7
    z0 = (project_fermion @ cvx_sigma.value @ project_fermion) / optim_value
    assert np.abs(z0-z0.T.conj()).max() < zero_eps
    assert abs(np.trace(z0)-1) < zero_eps
    assert np.linalg.eigvalsh(z0)[0]>-zero_eps
    tmp0 = z0.reshape(dim, dim, dim, dim).transpose(0,3,2,1).reshape(dim*dim, dim*dim)
    np.linalg.eigvalsh(tmp0)[0]
