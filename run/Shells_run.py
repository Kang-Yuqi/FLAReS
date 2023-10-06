import QingLongTool as ql
import matplotlib
import matplotlib.pyplot as plt
import os 
import numpy as np
import mpi4py.MPI as MPI
import healpy as hp

import configparser
config = configparser.ConfigParser()
config.read('run_config.ini')
root_unify = config['ROOT_PATH']['root_unify']
root_local_base = config['ROOT_PATH']['root_local_base']
root_cluster_base = config['ROOT_PATH']['root_cluster_base']

root_local = root_local_base + root_unify
root_cluster = root_cluster_base + root_unify

######################## load param ########################
param_path = os.path.join(root_local, 'ql_param.ini')
ql_param = ql.read_config(param_path)

######################## Nbody to density map ########################
comm = MPI.COMM_WORLD
comm_rank = comm.Get_rank()
comm_size = comm.Get_size()

boxsize_list = np.loadtxt(f"{os.path.join(root_local, f'nbody_sim/boxsize_list.txt')}")

for idx in range(len(boxsize_list)):

    if comm_rank == 0:
        os.makedirs(os.path.join(root_local, f'shells/nbody{idx+1}'), exist_ok=True)

    intervals = np.load(f"{os.path.join(root_local, f'shells/shell_intervals_b{idx+1}.npy')}")

    for counter in range(len(intervals)):
        box_base_name = os.path.join(ql_param['snapshot_root'], f'nbody{idx+1}/snapshot_{counter:03}.hdf5')
        if comm_rank == 0:
            print(box_base_name)
            _, _, _, _, _, _, redshift, _ = ql.lensing.box2des.readHeader(box_base_name)
            np.random.seed((idx+1)*(counter+1)*ql_param['rand_seed'])
            rtheta = np.random.uniform(low=0.0, high=2*np.pi, size=[3])
        else:
            rtheta = None
        rtheta = comm.bcast(rtheta, root=0)

        map_contrast_path = os.path.join(root_local, f'shells/nbody{idx+1}/contrast_map_{counter:03}.fits')

        if os.path.exists(map_contrast_path):
            if comm_rank == 0:
                map_contrast, header = hp.read_map(map_contrast_path, h=True)
        else:
            map_contrast = ql.lensing.box2des.singlebox2shell_project_mpi(comm, box_base_name, 
            intervals[counter], rotate_shell=rtheta, Nside_map=ql_param['Nside_map'])
            
        if comm_rank == 0:
            hp.fitsfunc.write_map(os.path.join(root_local, f'shells/nbody{idx+1}/contrast_map_{counter:03}.fits'), map_contrast,overwrite=True,extra_header=[('rotate_x', rtheta[0]),('rotate_y', rtheta[1]),('rotate_z', rtheta[2])])

            cl_matter_output = ql.lensing.analytical.Cl_matter([redshift],lmin=ql_param['lmin'],lmax=3*ql_param['Nside_map']-1,
                kmax=1e3,nonlinear=True,H0=ql_param['H0'], ombh2=ql_param['ombh2'], 
                omch2=ql_param['omch2'], tau=ql_param['tau'], mnu=ql_param['mnu'], omk=ql_param['omk'],
                As=ql_param['As'], ns=ql_param['ns'])

            cl_matter_linear_output = ql.lensing.analytical.Cl_matter([redshift],lmin=ql_param['lmin'],lmax=3*ql_param['Nside_map']-1,
                kmax=1e3,nonlinear=False,H0=ql_param['H0'], ombh2=ql_param['ombh2'], 
                omch2=ql_param['omch2'], tau=ql_param['tau'], mnu=ql_param['mnu'], omk=ql_param['omk'],
                As=ql_param['As'], ns=ql_param['ns'])

            ls,ks,cl_matter_nonlinear = cl_matter_output[0]
            ls,ks,cl_matter_linear = cl_matter_linear_output[0]

            cl = hp.sphtfunc.anafast(map_contrast)*np.mean(intervals[counter])**2*(intervals[counter][1]-intervals[counter][0])
            
            fig, ax1 = plt.subplots(figsize=(10, 6))
            ax1.loglog(ls, cl[ql_param['lmin']:], label='N-body')
            ax1.loglog(ls, cl_matter_nonlinear, label='CAMB Halofit')
            ax1.loglog(ls, cl_matter_linear, label='CAMB Linear')
            ax1.set_ylabel(r'$C_\ell~[(\mathrm{Mpc}/h)^3]$')
            ax1.set_xlabel(r'$\ell$')
            ax1.legend()

            num_ticks = 9
            min_log_l = 1 # Starting from the second value to avoid log(0)
            max_log_l = 3

            selected_log_ls = np.logspace(min_log_l, max_log_l, num_ticks)
            selected_indices = [np.abs(ls - l).argmin() for l in selected_log_ls]
            selected_ls = ls[selected_indices]
            selected_ks = ks[selected_indices]

            ax1.set_xticks(selected_ls)
            ax1.set_xticklabels([f"{int(l)}" for l in selected_ls])

            ax2 = ax1.twiny()
            ax2.set_xlim(ax1.get_xlim())
            ax2.set_xscale('log')
            ax2.set_xticks(selected_ls)
            ax2.set_xticklabels([f"{k:.3f}" for k in selected_ks])
            ax2.tick_params(which='minor', size=0)
            ax2.set_xlabel(r'$k$')

            ax1.set_title(r'matter power spectrum ($z$ = %.2f)' % redshift)
            ax1.grid()
            plt.tight_layout()
            plt.savefig(os.path.join(root_local, f'shells/nbody{idx+1}/contrast_map_Cl_{counter:03}.png'))
            plt.show()

            hp.visufunc.mollview(map=map_contrast)
            plt.savefig(os.path.join(root_local, f'shells/nbody{idx+1}/contrast_map_{counter:03}.png'))
            plt.show()

