import FlaresTool as fl
import os
import numpy as np
import healpy as hp
import camb
import readgadget
import mpi4py.MPI as MPI
######################## load param #########################
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-p', '--param-file', default='../param/param.ini', type=str, help='Path to the parameter file')

args = parser.parse_args()
fl_param = fl.read_config(args.param_file)
num_shells = fl_param['num_shells']
n = sum(num_shells)
N = fl_param['Gaussian_shell_steps']
lmax = fl_param['lmax']
lmin = fl_param['lmin']
h = fl_param['H0']/100
######################## path #########################
root_local = fl_param['root_local_base'] + fl_param['root_unify']
root_cluster = fl_param['root_cluster_base'] + fl_param['root_unify']
if fl_param['current_base'] == 'cluster':
    root_local = root_cluster
elif fl_param['current_base'] == 'local':
    pass 
else:
    raise TypeError("current_base should be 'local' or 'cluster'.")

######################## dirctories #########################
os.makedirs(os.path.join(root_local, 'lensing'), exist_ok=True)

######################## cosmology with CAMB ##################
pars = camb.CAMBparams()
pars.set_cosmology(H0=fl_param['H0'], ombh2=fl_param['ombh2'], omch2=fl_param['omch2'], 
    tau=fl_param['tau'], mnu=fl_param['mnu'], omk=fl_param['omk'])
pars.InitPower.set_params(As=fl_param['As'], ns=fl_param['ns'])
results = camb.get_results(pars)

## mpi setting
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if size > len(num_shells):
    tasks_per_rank = 1
else:
    tasks_per_rank = len(num_shells) // size
start_idx = rank * tasks_per_rank
end_idx = (rank + 1) * tasks_per_rank if rank != size - 1 else len(num_shells)


######################## shell thickness effect correction ##################
k_parallel_min = 0
k_parallel_max = 4

for idx in range(start_idx, end_idx):
    intervals = np.load(os.path.join(root_local, f'shells/shell_intervals_{n}Boxshells_b{idx+1}.npy'))/h
    chis = np.load(os.path.join(root_local, f'shells/shell_eff_midpoints_{n}Boxshells_b{idx+1}.npy'))/h
    redshifts = results.redshift_at_comoving_radial_distance(chis)

    PK = camb.get_matter_power_interpolator(pars, zs=redshifts, nonlinear=True, hubble_units=False, 
            k_hunit=False, kmax=10, var1=None, var2=None)
    ls = np.arange(lmax+1, dtype=np.float64)
    ratio_list = []
    for counter in range(len(intervals)):
        print(f"Box {idx}",f"shell {counter}")
        redshift = redshifts[counter]
        interval = intervals[counter]/h

        ks = (ls+0.5) / chis[counter]
        cl_th = PK.P(redshift, ks, grid=False)
        cl_th[:lmin] = 0

        cl_shell_predict = fl.shell_thickness_corr.compute_spectrum(lmin,lmax,redshift,PK,interval[1]-interval[0],chis[counter],k_parallel_min,k_parallel_max)

        cl_th[:lmin] = 1
        cl_shell_predict[:lmin] = 1
        ratio = cl_th/cl_shell_predict/(interval[1]-interval[0])
        ratio_list.append(ratio)

    np.save(os.path.join(root_local, f'shells/shell_thickness_corr_ratio_{n}Boxshells_lmax{lmax}_b{idx+1}.npy'),ratio_list)