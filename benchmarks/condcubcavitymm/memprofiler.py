import os, sys
import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt
from scipy.constants import c as c_light

sys.path.append('../../')

from solverFIT3D import SolverFIT3D
from gridFIT3D import GridFIT3D 
from wakeSolver import WakeSolver

# importing the library
from memory_profiler import profile

# instantiating the decorator
@profile
def script():
    # ---------- Domain setup ---------
    # Number of mesh cells
    Nx = 49+20
    Ny = 49+20
    Nz = 94+20
    #dt = 1.181512253e-12 # CST

    # Embedded boundaries
    stl_cavity = 'cavity.stl' 
    stl_shell = 'shell.stl'
    surf = pv.read(stl_shell)

    stl_solids = {'cavity': stl_cavity, 'shell': stl_shell}
    stl_materials = {'cavity': 'vacuum', 'shell': [10, 1.0, 10]}

    # Domain bounds
    xmin, xmax, ymin, ymax, zmin, zmax = surf.bounds
    Lx, Ly, Lz = (xmax-xmin), (ymax-ymin), (zmax-zmin)

    # set grid and geometry
    grid = GridFIT3D(xmin, xmax, ymin, ymax, zmin, zmax, Nx, Ny, Nz, 
                    stl_solids=stl_solids, 
                    stl_materials=stl_materials)
        
    # ------------ Beam source ----------------
    # Beam parameters
    sigmaz = 18.5e-3    #[m] -> 2 GHz
    q = 1e-9            #[C]
    beta = 1.0          # beam beta TODO
    xs = 0.             # x source position [m]
    ys = 0.             # y source position [m]
    xt = 0.             # x test position [m]
    yt = 0.             # y test position [m]
    # [DEFAULT] tinj = 8.53*sigmaz/c_light  # injection time offset [s] 

    # Simualtion
    wakelength = 1.e-3 #[m]
    add_space = 8   # no. cells

    wake = WakeSolver(q=q, sigmaz=sigmaz, beta=beta,
                xsource=xs, ysource=ys, xtest=xt, ytest=yt,
                add_space=add_space, save=True, logfile=True)

    # ----------- Solver & Simulation ----------
    # boundary conditions``
    bc_low=['pec', 'pec', 'pec']
    bc_high=['pec', 'pec', 'pec']

    solver = SolverFIT3D(grid, wake, #dt=dt,
                        bc_low=bc_low, bc_high=bc_high, 
                        use_stl=True, bg='pec',
                        verbose=1)
    # Plot settings
    if not os.path.exists('img/'): os.mkdir('img/')
    plotkw = {'title':'img/Ez', 
                'add_patch':'cavity', 'patch_alpha':0.3,
                'vmin':-1e4, 'vmax':1e4,
                'plane': [int(Nx/2), slice(0, Ny), slice(0+add_space, Nz-add_space)]}

    # Run wakefield time-domain simulation
    solver.wakesolve(wakelength=wakelength, add_space=add_space,
                    plot=False, plot_every=30, save_J=False,
                    **plotkw)

if __name__ == '__main__':
    
    script()
