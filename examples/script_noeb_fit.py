import numpy as np
from scipy.constants import c as c_light, epsilon_0 as eps_0, mu_0 as mu_0
import matplotlib
# matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib import patches
import os, sys
from tqdm import tqdm
from mpl_toolkits.axes_grid1 import make_axes_locatable

sys.path.append('../')

from solverFIT3D import SolverFIT3D
#from solver3D import EMSolver3D
from grid3D import Grid3D
from conductors3d import noConductor
from scipy.special import jv
from field import Field 

Z0 = np.sqrt(mu_0 / eps_0)

L = 1.
# Number of mesh cells
N = 6
Nx = N
Ny = N
Nz = N
Lx = L
Ly = L
Lz = L
dx = L / Nx
dy = L / Ny
dz = L / Nz

xmin = -Lx/2  + dx / 2
xmax = Lx/2 + dx / 2
ymin = - Ly/2 + dx / 2
ymax = Ly/2 + dx / 2
zmin = - Lz/2 + dx / 2
zmax = Lz/2 + dx / 2

conductors = noConductor()
bc_low=['Periodic', 'Periodic', 'Periodic']
bc_high=['Periodic', 'Periodic', 'Periodic']

#bc_low=['pec', 'pec', 'pec']
#bc_high=['pec', 'pec', 'pec']

NCFL = 0.5

# set FIT solver
gridFIT = Grid3D(xmin, xmax, ymin, ymax, zmin, zmax, Nx, Ny, Nz, conductors, 'FIT')
solverFIT = SolverFIT3D(gridFIT, bc_low=bc_low, bc_high=bc_high)

# set FDTD solver
#gridFDTD = Grid3D(xmin, xmax, ymin, ymax, zmin, zmax, Nx, Ny, Nz, conductors, 'FDTD')
#solverFDTD = EMSolver3D(gridFDTD, 'FDTD', NCFL, bc_low=bc_low, bc_high=bc_high)

# set source
#solverFDTD.Ez[xs, ys, zs] = 1.0*c_light
solverFIT.step_0 = False
Nt = 150
plane = 'YZ'

if plane == 'XY':
    x, y, z = slice(0,Nx), slice(0,Ny), int(Nz//2) #plane XY
    xs, ys, zs = int(3*Nx/4), int(3*Ny/4),  int(Nz/2)
    solverFIT.E[xs, ys, zs, 'z'] = 1.0*c_light
    title = '(x,y,Nz/2)'
    xax, yax = 'y', 'x'

if plane == 'YZ':
    x, y, z = int(Nx//2), slice(0,Ny), slice(0,Nz) #plane YZ
    xs, ys, zs = int(Nx/2), int(3*Ny/4),  int(3*Nz/4)
    solverFIT.E[xs, ys, zs, 'z'] = 1.0*c_light
    title = '(Nx/2,y,z)'
    xax, yax = 'z', 'y'

def plot_E_field(n, solverFIT, solverFDTD=None):

    if solverFDTD is not None: numplots = 2
    else: numplots = 1

    fig, axs = plt.subplots(numplots,3, tight_layout=True, figsize=[8,6])
    dims = {0:'x', 1:'y', 2:'z'}
    vmin, vmax = -1.e6, 1.e6
    #FIT
    extent = (0, N, 0, N)
    if solverFDTD is None: axx = axs
    else: axx = axs[0,:]

    for i, ax in enumerate(axx):
        #vmin, vmax = -np.max(np.abs(solverFIT.E[x, y, z, dims[i]])), np.max(np.abs(solverFIT.E[x, y, z, dims[i]]))
        im = ax.imshow(solverFIT.E[x, y, z, dims[i]], cmap='rainbow', vmin=vmin, vmax=vmax, extent=extent)
        fig.colorbar(im, cax=make_axes_locatable(ax).append_axes('right', size='5%', pad=0.05))
        ax.set_title(f'FIT E{dims[i]}{title}')
        ax.set_xlabel(xax)
        ax.set_ylabel(yax)
    
    #FDTD
    if solverFDTD is not None:
        ax = axs[1,0]
        extent = (0, N, 0, N)
        #vmin, vmax = -np.max(np.abs(solverFDTD.Ex[x, y, z])), np.max(np.abs(solverFDTD.Ex[x, y, z]))
        im = ax.imshow(solverFDTD.Ex[x, y, z], cmap='rainbow', vmin=vmin, vmax=vmax, extent=extent)
        fig.colorbar(im, cax=make_axes_locatable(ax).append_axes('right', size='5%', pad=0.05))
        ax.set_title(f'FDTD Ex{title}')
        ax.set_xlabel(xax)
        ax.set_ylabel(yax)

        ax = axs[1,1]
        extent = (0, N, 0, N)
        #vmin, vmax = -np.max(np.abs(solverFDTD.Ey[x, y, z])), np.max(np.abs(solverFDTD.Ey[x, y, z]))
        im = ax.imshow(solverFDTD.Ey[x, y, z], cmap='rainbow', vmin=vmin, vmax=vmax, extent=extent)
        fig.colorbar(im, cax=make_axes_locatable(ax).append_axes('right', size='5%', pad=0.05))
        ax.set_title(f'FDTD Ey{title}')
        ax.set_xlabel(xax)
        ax.set_ylabel(yax)

        ax = axs[1,2]
        extent = (0, N, 0, N)
        #vmin, vmax = -np.max(np.abs(solverFDTD.Ez[x, y, z])), np.max(np.abs(solverFDTD.Ez[x, y, z]))
        im = ax.imshow(solverFDTD.Ez[x, y, z], cmap='rainbow', vmin=vmin, vmax=vmax, extent=extent)
        fig.colorbar(im, cax=make_axes_locatable(ax).append_axes('right', size='5%', pad=0.05))
        ax.set_title(f'FDTD Ez{title}')
        ax.set_xlabel(xax)
        ax.set_ylabel(yax)
    

    fig.suptitle(f'E field, timestep={n}')
    fig.savefig('imgE/'+str(n).zfill(4)+'.png')
    plt.clf()
    plt.close(fig)

def plot_H_field(n, solverFIT, solverFDTD=None):
    if solverFDTD is not None: numplots = 2
    else: numplots = 1

    fig, axs = plt.subplots(numplots,3, tight_layout=True, figsize=[8,6])
    dims = {0:'x', 1:'y', 2:'z'}
    vmin, vmax = -5e3, 5e3
    extent = (0, N, 0, N)
    if solverFDTD is None: axx = axs
    else: axx = axs[0,:]
    
    #FIT
    for i, ax in enumerate(axx):
        #vmin, vmax = -np.max(np.abs(solverFIT.H[x, y, z, dims[i]])), np.max(np.abs(solverFIT.H[x, y, z, dims[i]]))
        im = ax.imshow(solverFIT.H[x, y, z, dims[i]], cmap='rainbow', vmin=vmin, vmax=vmax, extent=extent)
        fig.colorbar(im, cax=make_axes_locatable(ax).append_axes('right', size='5%', pad=0.05))
        ax.set_title(f'FIT H{dims[i]}{title}')
        ax.set_xlabel(xax)
        ax.set_ylabel(yax)

    #FDTD
    if solverFDTD is not None:
        ax = axs[1,0]
        #vmin, vmax = -np.max(np.abs(solverFDTD.Hx[x, y, z])), np.max(np.abs(solverFDTD.Hx[x, y, z]))
        im = ax.imshow(solverFDTD.Hx[x, y, z], cmap='rainbow', vmin=vmin, vmax=vmax, extent=extent)
        fig.colorbar(im, cax=make_axes_locatable(ax).append_axes('right', size='5%', pad=0.05))
        ax.set_title(f'FDTD Hx{title}')
        ax.set_xlabel(xax)
        ax.set_ylabel(yax)

        ax = axs[1,1]
        #vmin, vmax = -np.max(np.abs(solverFDTD.Hy[x, y, z])), np.max(np.abs(solverFDTD.Hy[x, y, z]))
        im = ax.imshow(solverFDTD.Hy[x, y, z], cmap='rainbow', vmin=vmin, vmax=vmax, extent=extent)
        fig.colorbar(im, cax=make_axes_locatable(ax).append_axes('right', size='5%', pad=0.05))
        ax.set_title(f'FDTD Hy{title}')
        ax.set_xlabel(xax)
        ax.set_ylabel(yax)

        ax = axs[1,2]
        #vmin, vmax = -np.max(np.abs(solverFDTD.Hz[x, y, z])), np.max(np.abs(solverFDTD.Hz[x, y, z]))
        im = ax.imshow(solverFDTD.Hz[x, y, z], cmap='rainbow', vmin=vmin, vmax=vmax, extent=extent)
        fig.colorbar(im, cax=make_axes_locatable(ax).append_axes('right', size='5%', pad=0.05))
        ax.set_title(f'FDTD Hz{title}')
        ax.set_xlabel(xax)
        ax.set_ylabel(yax)

    fig.suptitle(f'H field, timestep={n}')
    fig.savefig('imgH/'+str(n).zfill(4)+'.png')
    plt.clf()
    plt.close(fig)

for n in tqdm(range(Nt)):
    solverFIT.one_step()
    #solverFDTD.one_step()

plot_E_field(n, solverFIT)
plot_H_field(n, solverFIT)

