import numpy as np
import matplotlib.pyplot as plt
import sys, glob, os
from scipy.constants import c as c_light
from tqdm import tqdm

sys.path.append('../')

from solverFIT3D import SolverFIT3D
from gridFIT3D import GridFIT3D 
from sources import Dipole, PlaneWave

# ---------- Solver setup ---------
# Number of mesh cells
Nx = 3
Ny = 36
Nz = 48

# Domain bounds
xmin, xmax, ymin, ymax, zmin, zmax = [0., Nx, 0., Ny, 0., Nz]

# boundary conditions
bc_low=['periodic', 'periodic', 'pec']
bc_high=['periodic', 'periodic', 'pec']

# set FIT solver
grid = GridFIT3D(xmin, xmax, ymin, ymax, zmin, zmax, Nx, Ny, Nz)
solver = SolverFIT3D(grid, 
                     bc_low=bc_low, bc_high=bc_high, 
                     use_stl=False, bg='vacuum')
solver.step_0 = False

# Plot arguments
#from matplotlib.colors import LinearSegmentedColormap
#cmap = LinearSegmentedColormap.from_list('name', plt.cm.jet(np.linspace(0.1, 0.9))) # CST's colormap
cmap = 'bwr'

if not os.path.exists('img/'): os.mkdir('img/')
plotkw = {'title': 'img/frame', 
            'vmin':-1, 'vmax':1,
            'norm': 'linear',
            'cmap' : cmap,
            'plane': [int(Nx/2), slice(0, Ny), slice(0, Nz)], 
            'off_screen' : True,
            }

# Initial conditions
print('Simulating initial conditions:')
dip = {}
dip[0] = Dipole(zs=slice(5,6), ys=slice(15,16), nodes=10)
#dip[0] = Dipole(xs=1,ys=1,zs=1, nodes=10)
#dip[1] = Dipole(xs=1,ys=Ny-2,zs=1, nodes=10)
#dip[2] = Dipole(xs=1,ys=Ny-2,zs=Nz-2, nodes=10)
#dip[3] = Dipole(xs=1,ys=1,zs=Nz-2, nodes=10)

steps = 0
Nt = (solver.z.max()-solver.z.min())/2/c_light/solver.dt

for i in tqdm(range(int(Nt))):
    #source.update(solver, steps*solver.dt)
    for k in dip.keys():
        dip[k].update(solver, steps*solver.dt)

    solver.one_step()
    steps += 1

    if i%30 == 0:
        # plot and save frame
        solver.plot2D(field='E', component='z', n=steps, **plotkw)

'''
# Embedded boundaries from frames
path = '/mnt/h/user/e/edelafue/data/FIT/badapple/'
frames = sorted(glob.glob(path+'*.png'))[-10:]
from PIL import Image

for frame in tqdm(frames):

    print('frame: ')
    # load frame and generate 3d mask
    image = Image.open(frame) 
    imbw = np.flipud(np.array(image.convert('1'))) 
    #imbw = np.logical_not(imbw)
    mask = np.stack([imbw]*Nx)

    # update tensor
    solver.ieps = solver.ieps*mask
    solver.update_tensors(tensor='ieps')

    for i in range(10):
        #source.update(solver, steps*solver.dt)
        for k in dip.keys():
            dip[k].update(solver, steps*solver.dt)

        solver.one_step()
        steps += 1

    # Remove fields
    buffE = solver.E.toarray()
    buffH = solver.H.toarray()
    solver.E = solver.E * mask
    solver.H = solver.H * mask

    # plot and save frame
    solver.plot2D(field='E', component='z', n=steps, **plotkw)

    #restore fields
    solver.E.fromarray(buffE) 
    solver.H.fromarray(buffH)

'''
    



