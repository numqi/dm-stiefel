import os
import pickle
import numpy as np
import mpl_toolkits.mplot3d
import matplotlib.cm
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

import numqi

# plt.ion()

from utils import MagicStabilizerEntropyModel

if torch.get_num_threads()!=1:
    torch.set_num_threads(1)

hf_data = lambda *x: os.path.join('data', *x)

def plot_manifold_sine_sine():
    xdata = np.linspace(-np.pi, np.pi, 100)*1.1
    ydata = np.linspace(-np.pi, np.pi, 100)*1.1
    xdata,ydata = np.meshgrid(xdata, ydata, indexing='ij')
    zdata = np.sin(xdata)*np.sin(ydata)

    fig = plt.figure()
    ax = mpl_toolkits.mplot3d.Axes3D(fig, auto_add_to_figure=False)
    fig.add_axes(ax)
    hSurf = ax.plot_surface(xdata, ydata, zdata, cmap=plt.get_cmap('coolwarm'), linewidth=0, antialiased=True)
    ax.axis('off')
    ax.set_zlim(-3, 3)
    # ax.get_zlim()
    fig.savefig(hf_data('manifold-sine-sine.pdf'), transparent=True)


def demo_bloch_cross_section():
    datapath = 'data/bloch_cross_section.pkl'
    xlist = np.linspace(-0.54, 0.54, 101)
    ylist = np.linspace(-0.54, 0.54, 101)

    if not os.path.exists(datapath):
        model = MagicStabilizerEntropyModel(alpha=2, num_qubit=1, num_term=4)
        ret_opt = np.zeros((len(xlist)*len(ylist)), dtype=np.float64)
        # to completely remove those coarse points, a larger num_repeat is needed
        kwargs = dict(theta0='uniform', num_repeat=100, tol=1e-10, print_every_round=0, early_stop_threshold=(1e-8)-1)
        tmp0 = tuple((xi,yi) for xi in xlist for yi in ylist)
        for ind0,(xi,yi) in tqdm(enumerate(tmp0), total=len(xlist)*len(ylist)):
            tmp0 = np.array([xi, yi, 0])
            if np.linalg.norm(tmp0) > 0.5:
                ret_opt[ind0] = np.nan
            else:
                model.set_density_matrix(numqi.gellmann.gellmann_basis_to_dm(tmp0))
                ret_opt[ind0] = 1-(-numqi.optimize.minimize(model, **kwargs).fun)
        ret_opt = ret_opt.reshape(len(xlist), len(ylist))
        with open(datapath, 'wb') as fid:
            pickle.dump(dict(xlist=xlist, ylist=ylist, ret_opt=ret_opt), fid)
    else:
        with open(datapath, 'rb') as fid:
            tmp0 = pickle.load(fid)
            xlist = tmp0['xlist']
            ylist = tmp0['ylist']
            ret_opt = tmp0['ret_opt']

    tmp0 = ret_opt[np.logical_not(np.isnan(ret_opt))]
    ret_opt_min = tmp0.min()
    ret_opt_max = tmp0.max()
    print(ret_opt_min, ret_opt_max)
    # plt.get_cmap('winter')(np.array([0.0, 1.0])) #RGBA

    z0 = ret_opt.copy()
    z0[np.isnan(z0)] = ret_opt_min
    z0 = np.clip(z0, 1e-7, z0.max())
    tmp0 = (xlist.reshape(-1,1)**2 + ylist.reshape(1,-1)**2) > 0.255
    z0[tmp0] = np.nan
    fig,ax = plt.subplots()
    hcontourf = ax.contourf(xlist, ylist, np.log10(z0.T), levels=20, cmap='RdPu')
    tmp0 = np.linspace(0, 2*np.pi, 100)
    ax.plot(0.5*np.cos(tmp0), 0.5*np.sin(tmp0), linestyle='solid', color='black', linewidth=3)
    ax.set_xlim(-0.52, 0.52)
    ax.set_ylim(-0.52, 0.52)
    ax.set_aspect('equal')
    cax = fig.colorbar(hcontourf, shrink=0.8)
    cax.ax.get_yticks()
    tmp0 = list(range(-6, 0))
    cax.ax.set_yticks(tmp0)
    cax.ax.set_yticklabels(['$10^{}$'.format('{'+str(x)+'}') for x in tmp0])
    ax.axis('off')
    fig.tight_layout()
    fig.savefig('data/bloch_cross_section.png', dpi=200)
    fig.savefig('data/bloch_cross_section.pdf')


def get_cmap_hf(a=-7.5, b=0):
    # [a,b]->[0,1]
    hf_v2c = lambda x: (np.log10(x)-a)/(b-a) #[10^a,10^b]->[0,1]
    hf_c2v = lambda c: 10**(a+c*(b-a)) #[0,1]->[10^a,10^b]
    return hf_v2c, hf_c2v


def demo_ugly_3d_surface():
    datapath = 'data/bloch_sphere_magic.pkl'
    if os.path.exists(datapath):
        with open(datapath, 'rb') as fid:
            all_data = pickle.load(fid)
    else:
        #face1
        all_data = dict()
        r_list = np.linspace(0.1, 0.5, 41)
        t_list = np.linspace(0, 2*np.pi, 201)
        tmp0 = np.array([[[r*np.cos(t),r*np.sin(t),-np.sqrt(max(0,0.25-r*r))] for t in t_list] for r in r_list])
        tmp1 = numqi.gellmann.gellmann_basis_to_dm(tmp0)
        all_data['face1'] = dict(r_list=r_list, t_list=t_list, dm_list=tmp1)

        #face2
        r_list = np.linspace(0.3, 0.5, 41)
        t_list = np.linspace(0, 2*np.pi, 201)
        tmp0 = np.array([[[r*np.cos(t),r*np.sin(t),0] for t in t_list] for r in r_list])
        tmp1 = numqi.gellmann.gellmann_basis_to_dm(tmp0)
        all_data['face2'] = dict(r_list=r_list, t_list=t_list, dm_list=tmp1)

        #face3
        t_list = np.linspace(0, np.pi/2, 101)
        p_list = np.linspace(0, 2*np.pi, 201)
        tmp0 = 0.5*np.array([[[np.sin(t)*np.cos(p),np.sin(t)*np.sin(p),np.cos(t)] for p in p_list] for t in t_list])
        tmp1 = numqi.gellmann.gellmann_basis_to_dm(tmp0)
        boundary_list = numqi.magic.get_magic_state_boundary_qubit(tmp1.reshape(-1,2,2)).reshape(len(t_list),len(p_list))
        tmp2 = np.asarray([[numqi.entangle.hf_interpolate_dm(tmp1[x,y],beta=boundary_list[x,y]) for y in range(len(p_list))] for x in range(len(t_list))])
        all_data['face3'] = dict(t_list=t_list, p_list=p_list, dm_list=tmp2, boundary_list=boundary_list)


        model = MagicStabilizerEntropyModel(alpha=2, num_qubit=1, num_term=4)
        # to completely remove those coarse points, a larger num_repeat is needed
        kwargs = dict(theta0='uniform', num_repeat=100, tol=1e-10, print_every_round=0, early_stop_threshold=(1e-8)-1)
        for key in ['face1','face2','face3']:
            dm_list = all_data[key]['dm_list']
            shape = dm_list.shape[:2]
            tmp0 = tuple((x,y) for x in range(shape[0]) for y in range(shape[1]))
            tmp1 = []
            for ind0,ind1 in tqdm(tmp0, total=shape[0]*shape[1], desc=key):
                model.set_density_matrix(dm_list[ind0,ind1])
                tmp1.append(1-(-numqi.optimize.minimize(model, **kwargs).fun))
            all_data[key]['magic'] = np.array(tmp1).reshape(shape[0],shape[1])

        with open(datapath, 'wb') as fid:
            pickle.dump(all_data, fid)


    fig = plt.figure()
    ax = mpl_toolkits.mplot3d.Axes3D(fig, rect=[0.05,0.05,0.9,0.9], auto_add_to_figure=False)
    fig.add_axes(ax)
    cmap = plt.get_cmap('RdPu')
    hf_v2c,hf_c2v = get_cmap_hf(a=-7.5, b=-0.5)
    #face1
    r_list = all_data['face1']['r_list']
    t_list = all_data['face1']['t_list']
    magic_list = np.clip(all_data['face1']['magic'], 1e-7, 1)
    tmp0 = r_list.reshape(-1,1)*np.cos(t_list)
    tmp1 = r_list.reshape(-1,1)*np.sin(t_list)
    tmp2 = -np.sqrt(np.maximum(0,0.25-tmp0*tmp0 - tmp1*tmp1))
    # , rcount=100, ccount=100
    ax.plot_surface(tmp0, tmp1, tmp2, facecolors=cmap(hf_v2c(magic_list)), linewidth=0, edgecolor='none', antialiased=False, rcount=40, ccount=100)
    #face2
    r_list = all_data['face2']['r_list']
    t_list = all_data['face2']['t_list']
    magic_list = np.clip(all_data['face2']['magic'], 1e-7, 1)
    tmp0 = r_list.reshape(-1,1)*np.cos(t_list)
    tmp1 = r_list.reshape(-1,1)*np.sin(t_list)
    tmp2 = np.zeros_like(tmp0)
    ax.plot_surface(tmp0, tmp1, tmp2, facecolors=cmap(hf_v2c(magic_list)), linewidth=0, edgecolor='none', antialiased=False, rcount=40, ccount=100)
    #face3
    t_list = all_data['face3']['t_list']
    p_list = all_data['face3']['p_list']
    magic_list = np.clip(all_data['face3']['magic'], 1e-7, 1)
    ind0 = p_list <= np.pi
    tmp0,tmp1,tmp2 = numqi.gellmann.dm_to_gellmann_basis(all_data['face3']['dm_list']).transpose(2,0,1)
    # ax.plot_surface(tmp0[:,~ind0], tmp1[:,~ind0], tmp2[:,~ind0], facecolors=cmap(hf_v2c(magic_list))[:,:,:3], linewidth=0, edgecolor='none', antialiased=False)
    # ax.plot_surface(tmp0[:,ind0], tmp1[:,ind0], tmp2[:,ind0], facecolors=cmap(hf_v2c(magic_list))[:,:,:3], linewidth=0, edgecolor='none', antialiased=False)

    ax.set_xlim(-0.52, 0.52)
    ax.set_ylim(-0.52, 0.52)
    ax.set_zlim(-0.52, 0.52)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_aspect('equal')
    cax = fig.add_axes([0.9,0.05,0.02,0.9])
    fig.colorbar(matplotlib.cm.ScalarMappable(cmap=cmap), cax=cax)
    tmp0 = list(range(-6, 0))
    cax.set_yticks(hf_v2c(10.0**np.array(tmp0)))
    cax.set_yticklabels(['$10^{}$'.format('{'+str(x)+'}') for x in tmp0])
    ax.axis('off')
    fig.savefig('tbd01.png', dpi=200)


def quick_solve(dm_list, model, kwargs):
    assert (dm_list.ndim==3) and dm_list.shape[1:]==(2,2)
    ret = []
    for dm_i in tqdm(dm_list):
        model.set_density_matrix(dm_i)
        ret.append(1-(-numqi.optimize.minimize(model, **kwargs).fun))
    ret = np.array(ret)
    return ret


def plot_bloch_cross_section_i(key:str):
    assert key in {'origin','001','111'}
    if key=='111':
        extreme_point_list = np.array([[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]])/2
        datapath = 'data/bloch_cross_section_111.pkl'
        tmp0 = extreme_point_list[[0,2,5]]
        hf0 = lambda x: x/np.linalg.norm(x)
        xcenter = tmp0.mean(axis=0)
        basis0 = hf0(tmp0[1] - tmp0[0])
        basis1 = hf0(tmp0[0]+tmp0[1]-2*tmp0[2])
        cmap = 'YlOrBr'
        alpha = 0.7
    elif key=='001':
        datapath = 'data/bloch_cross_section_001.pkl'
        xcenter = np.array([0,0,0.25])
        basis0 = np.array([1,0,0])
        basis1 = np.array([0,1,0])
        cmap = 'Greys'
        alpha = 0.9
    elif key=='origin':
        datapath = 'data/bloch_cross_section_origin.pkl'
        xcenter = np.array([0,0,0])
        basis0 = np.array([1,0,0])
        basis1 = np.array([0,1,0])
        cmap = 'Blues'
        alpha = 0.9
    scale = 1.04
    num_point = 151
    if os.path.exists(datapath):
        with open(datapath, 'rb') as fid:
            tmp0 = pickle.load(fid)
            xlist = tmp0['xlist']
            ylist = tmp0['ylist']
            ret = tmp0['ret']
        pass
    else:
        hf0 = lambda x,y: abs(np.dot(x,y)) < 1e-10
        assert hf0(xcenter, basis0) and hf0(xcenter, basis1) and hf0(basis0, basis1)
        assert np.linalg.norm(xcenter) < 0.5
        tmp0 = np.sqrt(np.maximum(0, 0.25 - np.dot(xcenter, xcenter)))*scale
        xlist = np.linspace(-tmp0, tmp0, num_point)
        ylist = xlist

        tmp0 = np.asarray([[xcenter+x*basis0+y*basis1 for y in ylist] for x in xlist])
        mask = np.linalg.norm(tmp0, axis=2) < 0.5
        dm_list = numqi.gellmann.gellmann_basis_to_dm(tmp0)
        model = MagicStabilizerEntropyModel(alpha=2, num_qubit=1, num_term=4)
        # to completely remove those coarse points, a larger num_repeat is needed
        kwargs = dict(theta0='uniform', num_repeat=100, tol=1e-10, print_every_round=0, early_stop_threshold=(1e-8)-1)
        ret = np.zeros((len(xlist), len(ylist)), dtype=np.float64)
        tmp0 = [(x,y) for x in range(len(xlist)) for y in range(len(ylist))]
        for ind0,ind1 in tqdm(tmp0):
            if mask[ind0,ind1]:
                model.set_density_matrix(dm_list[ind0,ind1])
                ret[ind0,ind1] = 1-(-numqi.optimize.minimize(model, **kwargs).fun)
            else:
                ret[ind0,ind1] = np.nan
        with open(datapath, 'wb') as fid:
            pickle.dump(dict(xlist=xlist, ylist=ylist, ret=ret), fid)

    fig,ax = plt.subplots()
    tmp0 = np.log10(np.clip(ret.T, 1e-7, ret[~np.isnan(ret)].max()))
    hcontourf = ax.contourf(xlist, ylist, tmp0, alpha=alpha, levels=20, cmap=cmap)
    tmp0 = np.linspace(0, 2*np.pi, 100)
    tmp1 = np.sqrt(np.maximum(0, 0.25 - np.dot(xcenter, xcenter)))
    ax.plot(tmp1*np.cos(tmp0), tmp1*np.sin(tmp0), linestyle='solid', color='black', linewidth=3)
    ax.set_xlim(-0.52, 0.52)
    ax.set_ylim(-0.52, 0.52)
    ax.set_aspect('equal')
    cax = fig.colorbar(hcontourf, shrink=0.8)
    cax.ax.get_yticks()
    tmp0 = list(range(-6, 0))
    cax.ax.set_yticks(tmp0)
    cax.ax.set_yticklabels(['$10^{}$'.format('{'+str(x)+'}') for x in tmp0])
    ax.axis('off')
    fig.tight_layout()
    # fig.savefig('tbd01.png', dpi=200)
    fig.savefig(datapath.replace('.pkl','.png'), dpi=200)
    fig.savefig(datapath.replace('.pkl','.pdf'), transparent=True)


if __name__ == '__main__':
    plot_bloch_cross_section_i('origin')
    plot_bloch_cross_section_i('001')
    plot_bloch_cross_section_i('111')
