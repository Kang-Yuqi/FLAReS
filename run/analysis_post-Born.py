import FlaresTool as fl
import matplotlib.pyplot as plt
import os 
import numpy as np
import healpy as hp
import camb
from mpi4py import MPI

######################## load param #########################
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-p', '--param-file', default='../param/param.ini', type=str, help='Path to the parameter file')
parser.add_argument('-s', '--random-seed', default='0', type=int, help='random seed for shells rotation')


#################### analysis ####################

def calculate_l_corr(fl_param):
    Nside_map, Nside_shell, num_shells, n, N, lmax, lmin, lmax_analysis_ini, h, root_local, root_cluster,mixed_Nbody_folder_list,mixed_Nbody_suffix = fl.read_param(fl_param).values()

    boxsize_list = np.loadtxt(os.path.join(root_local, f"{mixed_Nbody_folder_list[0]}/boxsize_list.txt"))
    l_corr_list = []
    for idx in range(1, len(num_shells)):
        intervals = np.load(os.path.join(root_local, f'shells/shell_intervals_{n}Boxshells_b{idx+1}.npy'))
        midpoints = np.load(os.path.join(root_local, f'shells/shell_eff_midpoints_{n}Boxshells_b{idx+1}.npy'))
        l_corr = np.int_((2 * np.pi) * midpoints / (boxsize_list[idx] / 2) - 0.5)
        l_corr_list.extend(l_corr)
    max_l_corr = max(l_corr_list)
    min_l_corr = min(l_corr_list)
    return max_l_corr, min_l_corr

def plot_kappa_cl(fl_param,kappa_cl_list,kappa_names, mpi, rank, lmin=2, lmax=None, ls=None, save_path=None):

    if not mpi or (mpi and rank == 0):
        max_l_corr, min_l_corr = calculate_l_corr(fl_param)

        if ls is None:
            ls = np.arange(len(kappa_cl_list[0]))
        if lmax is None:
            lmax = len(ls) - 1
        if max_l_corr is not None and min_l_corr is not None:
            plt.axvline(x=max_l_corr, color='gray',linestyle='--')
            plt.axvline(x=min_l_corr, color='gray',linestyle='--')

        for i in range(len(kappa_cl_list)-1):
            plt.loglog(ls[lmin+1:lmax+1], kappa_cl_list[i][lmin+1:lmax+1], label=f'{kappa_names[i]}')
        plt.loglog(ls[lmin+1:lmax+1], kappa_cl_list[-1][lmin+1:lmax+1],linestyle='--', color='black', label=f'{kappa_names[-1]}')

        plt.ylabel(r'$C_\ell^{\kappa\kappa}$')
        plt.xlabel(r'$\ell$')
        plt.legend()
        plt.savefig(save_path)
        plt.close()
        print(f"Saved Cl plot to {save_path}")

def plot_kappa_map(fl_param, kappa_path_list, kappa_names_list,norm=None, save_path=None):

    Nside_map, Nside_shell, num_shells, n, N, lmax, lmin, lmax_analysis_ini, h, root_local, root_cluster, mixed_Nbody_folder_list, mixed_Nbody_suffix = fl.read_param(fl_param).values()
    num_plots = len(kappa_names_list)

    for idx in range(num_plots):
        if isinstance(kappa_path_list[idx], list):
            kappa_map_sum = hp.fitsfunc.read_map(kappa_path_list[idx][0])
            for path in kappa_path_list[idx][1:]:
                kappa_map_sum += hp.fitsfunc.read_map(path)
            kappa_map = kappa_map_sum
        else:
            kappa_map = hp.fitsfunc.read_map(kappa_path_list[idx])

        hp.projview(kappa_map,
            title=f"$\kappa$ {kappa_names_list[idx]}",
            width=10,
            norm=norm,
            sub=(num_plots//2, 2, idx + 1),)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    print(f"Saved kappa map plot to {save_path}")

def theoretical_cl(fl_param,pars=None,save_dir=None,lmax_analysis=None,new=False):
    file_path = os.path.join(save_dir, f"cl_Born_theory.txt")
    if not os.path.exists(file_path) or new:
        print('### Calculatiing theoretical power spectrum')
        Nside_map, Nside_shell, num_shells, n, N, lmax, lmin, lmax_analysis_ini, h, root_local, root_cluster,mixed_Nbody_folder_list,mixed_Nbody_suffix = fl.read_param(fl_param).values()
        if lmax_analysis is None:
            lmax_analysis = lmax_analysis_ini
        if pars is None:
            pars = camb.CAMBparams()
            pars.set_cosmology(H0=fl_param['H0'], ombh2=fl_param['ombh2'], omch2=fl_param['omch2'], 
                tau=fl_param['tau'], mnu=fl_param['mnu'], omk=fl_param['omk'])
            pars.InitPower.set_params(As=fl_param['As'], ns=fl_param['ns'])

        [ls,cl_kappa_th] = fl.analytical_power_spectrum.lensing_kappa_Cl_range(pars,num_split=200,lmin=lmin,lmax = lmax_analysis,z_low=None,z_high=None)
        np.savetxt(file_path,[ls,cl_kappa_th])
        return [ls,cl_kappa_th]
    else:
        print('### Theoretical Cl^{kappa kappa} exist.')
        [ls,cl_kappa_th] = np.loadtxt(file_path)
        return [ls,cl_kappa_th]

def kappa_cl_calc(fl_param, map_path=None, save_dir=None, type_label=None, lmax_analysis=None, new=False):

    Nside_map, Nside_shell, num_shells, n, N, lmax, lmin, lmax_analysis_ini, h, root_local, root_cluster, mixed_Nbody_folder_list, mixed_Nbody_suffix = fl.read_param(fl_param).values()
    file_path = os.path.join(save_dir, f"cl_{type_label}_{n}Boxshells_{N}Gshells_{mixed_Nbody_suffix}_seed{random_seed}.txt")

    if os.path.exists(file_path) and not new:
        print(f'### kappa {type_label} map power spectrum exists.')
        cl_kappa = np.loadtxt(file_path)
        return cl_kappa

    print(f'### Calculating kappa {type_label} map power spectrum.')

    if isinstance(map_path, list):
        kappa_map_sum = hp.fitsfunc.read_map(map_path[0])
        for path in map_path[1:]:
            kappa_map_sum += hp.fitsfunc.read_map(path)
        kappa_map = kappa_map_sum
    else:
        kappa_map = hp.fitsfunc.read_map(map_path)

    if lmax_analysis is None:
        lmax_analysis = lmax_analysis_ini
    cl_kappa = hp.sphtfunc.anafast(kappa_map, lmax=lmax_analysis)

    np.savetxt(file_path, cl_kappa)

    return cl_kappa

def cl_calculations_mpi(fl_param, kappa_names, kappa_paths, lmax_analysis, save_dir=None, pars=None, new=False):

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    kappa_cl_list = []
    num_tasks = len(kappa_names)
    for idx in range(rank, num_tasks, size):
        cl_kappa = kappa_cl_calc(fl_param, map_path=kappa_paths[idx], save_dir=save_dir, type_label=kappa_names[idx], lmax_analysis=lmax_analysis, new=new)
        kappa_cl_list.append((idx, cl_kappa))

    all_kappa_cl_list = comm.gather(kappa_cl_list, root=0)
    final_kappa_cl_list = None
    if rank == 0:
        final_kappa_cl_list = [None] * num_tasks
        for kappa_cl in all_kappa_cl_list:
            for idx, cl in kappa_cl:
                final_kappa_cl_list[idx] = cl
        [ls, cl_kappa_th] = theoretical_cl(fl_param, pars=pars, save_dir=save_dir, lmax_analysis=lmax_analysis, new=new)
        final_kappa_cl_list.append(cl_kappa_th)
        kappa_names.append("theoretical")
    comm.Barrier()
    return final_kappa_cl_list, kappa_names if rank == 0 else (None, None)

def cl_calculations(fl_param, kappa_names, kappa_paths, lmax_analysis, save_dir=None, pars=None, new=False):

    kappa_cl_list = []
    for idx in range(len(kappa_names)):
        cl_kappa = kappa_cl_calc(fl_param,map_path=kappa_paths[idx],save_dir=save_dir,type_label=kappa_names[idx],lmax_analysis=lmax_analysis,new=False)
        kappa_cl_list.append(cl_kappa)

    [ls,cl_kappa_th] = theoretical_cl(fl_param,pars=pars,save_dir=save_dir,lmax_analysis=lmax_analysis,new=False)
    kappa_names.append("theoretical")
    kappa_cl_list.append(cl_kappa_th)

    return kappa_cl_list, kappa_names

def NG_calculations(kappa_path, lmax_analysis, Nside_map, save_path,if_needlet,B,j):

    if if_needlet:
        if isinstance(kappa_path, list):
            kappa_map_sum = hp.fitsfunc.read_map(kappa_path[0])
            for path in kappa_path[1:]:
                kappa_map_sum += hp.fitsfunc.read_map(path)
            kappa_map = kappa_map_sum
        else:
            kappa_map = hp.fitsfunc.read_map(kappa_path)

        alm = hp.sphtfunc.map2alm(kappa_map, lmax=lmax_analysis)
        _, alm_filter = fl.needlets.needlet_alm(alm, j, B=B, lmax=lmax_analysis)
        nG_list = fl.nG_analysis.S_K_parameter_var(alm_filter, Nside_map, lmax_analysis)
        np.savetxt(save_path, nG_list)
        return nG_list   
    else:
        if isinstance(kappa_path, list):
            kappa_map_sum = hp.fitsfunc.read_map(kappa_path[0])
            for path in kappa_path[1:]:
                kappa_map_sum += hp.fitsfunc.read_map(path)
            kappa_map = kappa_map_sum
        else:
            kappa_map = hp.fitsfunc.read_map(kappa_path)
        alm = hp.sphtfunc.map2alm(kappa_map, lmax=lmax_analysis)
        nG_list = fl.nG_analysis.S_K_parameter_var(alm, Nside_map, lmax_analysis)
        np.savetxt(save_path, nG_list)
        return nG_list

def kappa_analysis_single(fl_param,pars=None, power_spectrum=True, plot_map=True,map_color_norm=None, NG=True, if_needlet=True, B=3.1, js=[2,3,4,5,6,7], lmax_analysis=None,mpi=True, random_seed=0):
    if mpi:
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
    else:
        rank = 0
        size = 1

    Nside_map, Nside_shell, num_shells, n, N, lmax, lmin, lmax_analysis_ini, h, root_local, root_cluster,mixed_Nbody_folder_list,mixed_Nbody_suffix = fl.read_param(fl_param).values()
    if lmax_analysis is None:
        lmax_analysis_list = [lmax_analysis_ini]
    elif isinstance(lmax_analysis, (int, float)):
        lmax_analysis_list = [int(lmax_analysis)]
    elif isinstance(lmax_analysis, list):
        lmax_analysis_list = [int(l) for l in lmax_analysis]

    lensing_path = os.path.join(root_local, f"lensing/lensing-Nside{Nside_map}_shell-Nside{Nside_shell}_lmax{lmax}")
    if not os.path.exists(lensing_path):
        fl.print_message(f'The lensing folder {lensing_path} not exists',mpi)
        return

    file_path_B_map = os.path.join(lensing_path, f"map_kappa_{n}Boxshells_{N}Gshells_{mixed_Nbody_suffix}_lensingNside{Nside_map}_lmax{lmax}_seed{random_seed}.fits")
    file_path_ll_map = os.path.join(lensing_path, f"map_kappa_ll_{n}Boxshells_{N}Gshells_{mixed_Nbody_suffix}_lensingNside{Nside_map}_lmax{lmax}_seed{random_seed}.fits")
    file_path_geo_map = os.path.join(lensing_path, f"map_kappa_geo_{n}Boxshells_{N}Gshells_{mixed_Nbody_suffix}_lensingNside{Nside_map}_lmax{lmax}_seed{random_seed}.fits")

    kappa_names_list = []
    kappa_path_list = []
    if os.path.exists(file_path_B_map) and os.path.exists(file_path_ll_map) and os.path.exists(file_path_geo_map):
        kappa_names_list.append("Born_post-Born")
        kappa_path_list.append([file_path_B_map,file_path_ll_map,file_path_geo_map])

    if os.path.exists(file_path_B_map):
        kappa_names_list.append("Born")
        kappa_path_list.append(file_path_B_map)
    if os.path.exists(file_path_ll_map):
        kappa_names_list.append("lens-lens")
        kappa_path_list.append(file_path_ll_map)
    if os.path.exists(file_path_geo_map):
        kappa_names_list.append("geodesic")
        kappa_path_list.append(file_path_geo_map)

    root_analysis = os.path.join(root_local, 'analysis')

    if power_spectrum:
        kappa_cl_names_list = kappa_names_list.copy()
        lensing_analysis_data_path = os.path.join(root_analysis, f"lensing/data/lensing-Nside{Nside_map}_shell-Nside{Nside_shell}_lmax{lmax}_lmax_analysis{np.max(lmax_analysis_list)}")
        os.makedirs(lensing_analysis_data_path, exist_ok=True)
        if mpi:
            all_kappa_cl_list, all_kappa_names_list = cl_calculations_mpi(fl_param, kappa_cl_names_list, kappa_path_list, np.max(lmax_analysis_list), save_dir=lensing_analysis_data_path, pars=pars, new=True)
        else:
            all_kappa_cl_list, all_kappa_names_list = cl_calculations(fl_param, kappa_cl_names_list, kappa_path_list, np.max(lmax_analysis_list), save_dir=lensing_analysis_data_path, pars=pars, new=True)

        save_path = os.path.join(lensing_path, f"cl_kappa_{n}Boxshells_{N}Gshells_{mixed_Nbody_suffix}_lensingNside{Nside_map}_lmax{np.max(lmax_analysis_list)}_seed{random_seed}.pdf")
        plot_kappa_cl(fl_param,all_kappa_cl_list, all_kappa_names_list, mpi, rank, lmin=lmin, lmax=np.max(lmax_analysis_list), ls=None, save_path=save_path)

    if plot_map:
        if not mpi or (mpi and rank == 0):
            save_path = os.path.join(lensing_path, f"map_kappa_{n}Boxshells_{N}Gshells_{mixed_Nbody_suffix}_lensingNside{Nside_map}_lmax{lmax}_seed{random_seed}.pdf")
            plot_kappa_map(fl_param, kappa_path_list, kappa_names_list, norm=map_color_norm, save_path=save_path)

    if NG:
        tasks = []
        for lmax_analysis in lmax_analysis_list:
            lensing_analysis_data_path = os.path.join(root_analysis, f"lensing/data/lensing-Nside{Nside_map}_shell-Nside{Nside_shell}_lmax{lmax}_lmax_analysis{lmax_analysis}")
            os.makedirs(lensing_analysis_data_path, exist_ok=True)
            if os.path.exists(file_path_B_map):
                save_path_born = os.path.join(lensing_analysis_data_path, f"nG_Born_{n}Boxshells_{N}Gshells_{mixed_Nbody_suffix}_seed{random_seed}.txt")
                tasks.append((file_path_B_map, lmax_analysis, Nside_map, save_path_born,False,None,None))
            if os.path.exists(file_path_B_map) and os.path.exists(file_path_ll_map) and os.path.exists(file_path_geo_map):
                save_path_combined = os.path.join(lensing_analysis_data_path, f"nG_Born_post-Born_{n}Boxshells_{N}Gshells_{mixed_Nbody_suffix}_seed{random_seed}.txt")
                tasks.append(([file_path_B_map, file_path_ll_map, file_path_geo_map], lmax_analysis, Nside_map, save_path_combined,False,None,None))
        
        if if_needlet:
            lensing_analysis_data_path = os.path.join(root_analysis, f"lensing/data/lensing-Nside{Nside_map}_shell-Nside{Nside_shell}_lmax{lmax}_lmax_analysis{np.max(lmax_analysis_list)}")
            os.makedirs(lensing_analysis_data_path, exist_ok=True)
            if os.path.exists(file_path_B_map):
                for j in js:
                    save_path_born = os.path.join(lensing_analysis_data_path, f"nG_Born_j{j}_{n}Boxshells_{N}Gshells_{mixed_Nbody_suffix}_seed{random_seed}.txt")
                    tasks.append((file_path_B_map, lmax_analysis, Nside_map, save_path_born,True,B,j))
            if os.path.exists(file_path_B_map) and os.path.exists(file_path_ll_map) and os.path.exists(file_path_geo_map):
                for j in js:
                    save_path_combined = os.path.join(lensing_analysis_data_path, f"nG_Born_post-Born_j{j}_{n}Boxshells_{N}Gshells_{mixed_Nbody_suffix}_seed{random_seed}.txt")
                    tasks.append(([file_path_B_map, file_path_ll_map, file_path_geo_map], lmax_analysis, Nside_map, save_path_combined,True,B,j))

        if mpi:
            num_tasks = len(tasks)
            for idx in range(rank, num_tasks, size):
                map_paths, lmax_analysis, Nside_map, save_path,if_needlet,B,j = tasks[idx]
                print('rank',rank,B,j)
                NG_calculations(map_paths, lmax_analysis, Nside_map, save_path,if_needlet,B,j)

        else:
            for idx in range(num_tasks):
                map_paths, lmax_analysis, Nside_map, save_path,if_needlet,B,j = tasks[idx]
                NG_calculations(map_paths, lmax_analysis, Nside_map, save_path,if_needlet,B,j)


################################################ run ################################################
args = parser.parse_args()
random_seed = args.random_seed
fl_param = fl.read_config(args.param_file)
mpi = True
######################## cosmology with CAMB ##################
pars = camb.CAMBparams()
pars.set_cosmology(H0=fl_param['H0'], ombh2=fl_param['ombh2'], omch2=fl_param['omch2'], 
    tau=fl_param['tau'], mnu=fl_param['mnu'], omk=fl_param['omk'])
pars.InitPower.set_params(As=fl_param['As'], ns=fl_param['ns'])
results = camb.get_results(pars)

kappa_analysis_single(fl_param,pars=pars, power_spectrum=True, plot_map=True, map_color_norm=None, NG=True, if_needlet=True, B=3.1, js=[2,3,4,5,6,7], lmax_analysis=None,mpi=mpi, random_seed=int(random_seed))