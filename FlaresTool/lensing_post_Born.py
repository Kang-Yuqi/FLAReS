import os 
import numpy as np
import healpy as hp
import camb
from astropy.constants import c
import astropy.units as u
from mpi4py import MPI

def read_param(fl_param):
    Nside_map = fl_param['Nside_lensing']
    Nside_shell = fl_param['Nside_shell']
    num_shells = fl_param['num_shells']
    n = sum(num_shells)
    N = fl_param['Gaussian_shell_steps']
    lmax = fl_param['lmax']
    lmin = fl_param['lmin']
    lmax_analysis = fl_param['lmax_analysis']
    h = fl_param['H0'] / 100

    root_local = fl_param['root_local_base'] + fl_param['root_unify']
    root_cluster = fl_param['root_cluster_base'] + fl_param['root_unify']
    
    if fl_param['current_base'] == 'cluster':
        root_local = root_cluster
    elif fl_param['current_base'] == 'local':
        pass
    else:
        raise TypeError("current_base should be 'local' or 'cluster'.")

    try:
        mixed_Nbody_folder_list = fl_param['mixed_Nbody_folder_list']
        try:
            mixed_Nbody_suffix = fl_param['mixed_Nbody_suffix']
        except KeyError:
            mixed_Nbody_suffix = f'mixed{len(mixed_Nbody_folder_list)}nbody'
    except KeyError:
        mixed_Nbody_folder_list = [fl_param['Nbody_folder']]
        mixed_Nbody_suffix = fl_param['Nbody_folder']

    # Return parameters as a dictionary
    return {
        'Nside_map': Nside_map,
        'Nside_shell': Nside_shell,
        'num_shells': num_shells,
        'n': n,
        'N': N,
        'lmax': lmax,
        'lmin': lmin,
        'lmax_analysis':lmax_analysis,
        'h': h,
        'root_local': root_local,
        'root_cluster': root_cluster,
        'mixed_Nbody_folder_list':mixed_Nbody_folder_list,
        'mixed_Nbody_suffix':mixed_Nbody_suffix
    }

def print_message(message,mpi):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    if mpi:
        if rank == 0:
            print(message)
    else:
        print(message)

def lensing_summation(fl_param,map_in, map_in_sum, redshift, chi, chistar, interval,om=None):
    H0 = fl_param['H0']
    factor = 3*H0**2*om/(2*c.to('km/s').value**2)
    map_out_sum = map_in_sum + factor*map_in*((chistar-chi)*chi/chistar)/(1/(1+redshift))*(interval[1]-interval[0])
    return map_out_sum

def large_scale_replace(l_corr,alm_contrast,alm_G,lmax,Nside_map=None,return_alm=False):
    fl_G = np.ones(lmax+1)
    fl_G[l_corr:]=0
    alm_G_filter = hp.sphtfunc.almxfl(alm_G, fl_G, inplace=False)

    fl_contrast = np.ones(lmax+1)
    fl_contrast[:l_corr]=0
    alm_contrast_filter = hp.sphtfunc.almxfl(alm_contrast, fl_contrast, inplace=False)

    alm_corr = alm_contrast_filter + alm_G_filter
    
    if return_alm:
        return alm_corr
    else:
        map_contrast = hp.sphtfunc.alm2map(alm_corr, Nside_map, lmax=lmax)
        return map_contrast

def rotate_map_pixel(map_contrast,seed):

    rot_gal2eq = hp.Rotator(coord="GC")
    np.random.seed(seed)
    longitude = np.random.uniform(-180, 180) * u.deg
    latitude = np.random.uniform(0, 180) * u.deg
    rot_custom = hp.Rotator(rot=[longitude.to_value(u.deg), latitude.to_value(u.deg)])
    map_contrast_rotated = rot_custom.rotate_map_pixel(map_contrast)

    return map_contrast_rotated

def rotate_map_alm(alm_contrast,seed,lmax):

    rot_gal2eq = hp.Rotator(coord="GC")
    np.random.seed(seed)
    longitude = np.random.uniform(-180, 180) * u.deg
    latitude = np.random.uniform(0, 180) * u.deg
    rot_custom = hp.Rotator(rot=[longitude.to_value(u.deg), latitude.to_value(u.deg)])
    alm_contrast_rotated = rot_custom.rotate_alm(alm_contrast, lmax=lmax, mmax=None, inplace=False)

    return alm_contrast_rotated

def prepare_Gaussian_lens_shell(fl_param,cache_dir,pars,redshift,midpoint_G,interval_G,shell_idx,random_seed,new=False):

    Nside_map, Nside_shell, num_shells, n, N, lmax, lmin, lmax_analysis, h, root_local, root_cluster,mixed_Nbody_folder_list,mixed_Nbody_suffix = read_param(fl_param).values()

    os.makedirs(os.path.join(cache_dir,f'shell_contrast'), exist_ok=True)
    file_path = os.path.join(cache_dir,f'shell_contrast/alm_contrast_{n}Boxshells_{N}Gshells_{mixed_Nbody_suffix}_lensingNside{Nside_map}_counter{shell_idx}.npy')

    if not os.path.exists(file_path) or new:
        cl_matter_nonlinear_list = analytical_power_spectrum.Cl_matter([redshift],pars,
            lmin=lmin,lmax=lmax,kmax=1e2,hubble_units=False,nonlinear=True)
        [ls,ks,cl_matter] = cl_matter_nonlinear_list[0]

        np.random.seed(random_seed+shell_idx*100)
        alm_contrast = hp.sphtfunc.synalm(cl_matter, lmax=lmax)/midpoint_G/(interval_G[1]-interval_G[0])**0.5

        np.save(os.path.join(cache_dir,f'shell_contrast/alm_contrast_{n}Boxshells_{N}Gshells_{mixed_Nbody_suffix}_lensingNside{Nside_map}_counter{shell_idx}.npy'),alm_contrast)
    else:
        print(f'shell {shell_idx} exist, skip...')

def prepare_Nbody_lens_shell(fl_param,cache_dir,pars,redshift,midpoint,interval,idx,counter,min_l_corr,task_id,boxsize_list,Nbody_folder,overall_counter,random_seed,new=False):

    Nside_map, Nside_shell, num_shells, n, N, lmax, lmin, lmax_analysis, h, root_local, root_cluster,mixed_Nbody_folder_list,mixed_Nbody_suffix = read_param(fl_param).values()

    os.makedirs(os.path.join(cache_dir,f'shell_contrast'), exist_ok=True)
    file_path = os.path.join(cache_dir,f'shell_contrast/alm_contrast_{n}Boxshells_{N}Gshells_{mixed_Nbody_suffix}_lensingNside{Nside_map}_counter{overall_counter}.npy')

    if not os.path.exists(file_path) or new:
        print('shell idx:', overall_counter,'Nbody_folder:',Nbody_folder,'shell midpoint=', midpoint,'interval=', interval,'redshift=', redshift)

        np.random.seed(random_seed+idx*100+counter*1000+10000)
        cl_matter_list = analytical_power_spectrum.Cl_matter([redshift],pars,lmin=lmin,lmax=lmax,kmax=1e2,hubble_units=False,nonlinear=True)
        [ls_matter,ks_matter,cl_matter] = cl_matter_list[0]
        alm_G = hp.sphtfunc.synalm(cl_matter, lmax=lmax)/midpoint/(interval[1]-interval[0])**0.5

        map_contrast = hp.fitsfunc.read_map(os.path.join(root_local, f"shells/nbody{idx+1}/contrast_map_{Nbody_folder}_Nside{Nside_shell}_{counter:03}.fits"))
        alm_contrast = hp.sphtfunc.map2alm(map_contrast, lmax=lmax)

        # shell thickness effct correction
        shell_corr_ratio_list = np.load(os.path.join(root_local, f'shells/shell_thickness_corr_ratio_{n}Boxshells_lmax{lmax}_b{idx+1}.npy'))
        shell_corr_ratio = shell_corr_ratio_list[counter]**0.5
        alm_contrast = hp.sphtfunc.almxfl(alm_contrast, shell_corr_ratio[:lmax+1], mmax=None, inplace=False)

        # replace large scale with Gaussian realization
        l_corr = int((2*np.pi)*midpoint/(boxsize_list[idx]/2/h)-0.5)
        if l_corr < min_l_corr:
            l_corr = min_l_corr
        alm_contrast = large_scale_replace(l_corr,alm_contrast,alm_G,lmax,Nside_map,return_alm=True)

        # rotate the contrast map
        alm_contrast = rotate_map_alm(alm_contrast,seed=random_seed+idx*100+counter*1000+2000,lmax=lmax)
        np.save(file_path,alm_contrast)
    else:
        print(f'shell {overall_counter} exist, skip...')

def contrast2potential(alm_contrast, redshift, interval=None, lmax=None, H0=None, om=None):
    ell = np.arange(lmax + 1)
    poisson_filter = np.zeros_like(ell, dtype=np.float64)
    poisson_filter[2:] = -1.0 / (ell[2:] * (ell[2:] + 1))
    factor = 3*H0**2*om/(2*c.to('km/s').value**2)
    alm_contrast_mod = factor*alm_contrast/(1/(1+redshift))
    alm_phi = hp.sphtfunc.almxfl(alm_contrast_mod, poisson_filter)
    return alm_phi

def second_partial_Phi(alm,nside=None,lmax=None):
    _,map_dt,map_dp = hp.sphtfunc.alm2map_der1(alm,nside)
    alm_dp = hp.sphtfunc.map2alm(map_dp, lmax=lmax, iter=5)
    alm_dt = hp.sphtfunc.map2alm(map_dt, lmax=lmax, iter=5)
    _,map_dpt,map_dpp = hp.sphtfunc.alm2map_der1(alm_dp,nside)
    _,map_dtt,map_dtp = hp.sphtfunc.alm2map_der1(alm_dt,nside)

    return [map_dtt,map_dpp,map_dtp,map_dpt]

def prepare_phi_d(fl_param,om,task_list,cache_dir,j,new=False):
    Nside_map, Nside_shell, num_shells, n, N, lmax, lmin, lmax_analysis, h, root_local, root_cluster,mixed_Nbody_folder_list,mixed_Nbody_suffix = read_param(fl_param).values()

    file_path = os.path.join(cache_dir, f"phi_d/phi_d_{j}.npz")
    redshift_j, interval_j, midpoint_j = task_list[j]
    if not os.path.exists(file_path) or new:
        print('### Calculating first partial Phi', j,' redshift=',redshift_j)
        alm_contrast_j = np.load(os.path.join(cache_dir, f'shell_contrast/alm_contrast_{n}Boxshells_{N}Gshells_{mixed_Nbody_suffix}_lensingNside{Nside_map}_counter{j}.npy'))
        _,contrast_dt,contrast_dp = hp.sphtfunc.alm2map_der1(alm_contrast_j,Nside_map)
        alm_phi_j = contrast2potential(alm_contrast_j, redshift_j, lmax=lmax, H0=fl_param['H0'], om=om)
        _,phi_dt,phi_dp = hp.sphtfunc.alm2map_der1(alm_phi_j,Nside_map)
        np.savez(file_path, contrast_dt=contrast_dt, contrast_dp=contrast_dp, phi_dt=phi_dt, phi_dp=phi_dp)
    else:
        print(f'### First partial Phi {j} exist, skip...')

def prepare_phi_dd(fl_param,om,task_list,cache_dir,j,new=False):
    Nside_map, Nside_shell, num_shells, n, N, lmax, lmin, lmax_analysis, h, root_local, root_cluster,mixed_Nbody_folder_list,mixed_Nbody_suffix = read_param(fl_param).values()

    file_path = os.path.join(cache_dir, f"phi_dd/phi_dd_{j}.npz")
    redshift_j, interval_j, midpoint_j = task_list[j]
    if not os.path.exists(file_path) or new:
        print('### Calculating second partial Phi', j,' redshift=',redshift_j)
        alm_contrast_j = np.load(os.path.join(cache_dir, f'shell_contrast/alm_contrast_{n}Boxshells_{N}Gshells_{mixed_Nbody_suffix}_lensingNside{Nside_map}_counter{j}.npy'))
        alm_phi_j = contrast2potential(alm_contrast_j, redshift_j, lmax=lmax, H0=fl_param['H0'], om=om)
        [phi_dtt, phi_dpp, phi_dtp, phi_dpt] = second_partial_Phi(alm_phi_j, nside=Nside_map, lmax=lmax)
        np.savez(file_path, phi_dtt=phi_dtt, phi_dpp=phi_dpp, phi_dtp=phi_dtp, phi_dpt=phi_dpt)
    else:
        print(f'### Second partial Phi {j} exist, skip...')

def geo_sum(fl_param,task_list,cache_dir,j,om,Nside_map):

    print(f'### post-Born lens summation contrast {j} ...')

    _, _, midpoint_j = task_list[j]
    contrast_dt_map_sum = np.zeros(Nside_map**2 * 12)
    contrast_dp_map_sum = np.zeros(Nside_map**2 * 12)
    for j_inner in range(j):
        file_path_j_inner = os.path.join(cache_dir, f"phi_d/phi_d_{j_inner}.npz")
        redshift_j_inner, interval_j_inner, midpoint_j_inner = task_list[j_inner]

        if os.path.exists(file_path_j_inner):
            try:
                data_j_inner = np.load(file_path_j_inner)
                contrast_j_inner_dt, contrast_j_inner_dp = data_j_inner['contrast_dt'], data_j_inner['contrast_dp']
            except Exception as e:
                print(f"Error loading file {file_path_j_inner}.")
        else:
            print(f'### First partial contrast {j_inner} not exist.')
            prepare_phi_d(fl_param,om,task_list,cache_dir,j)
            data_j_inner = np.load(file_path_j_inner)
            contrast_j_inner_dt, contrast_j_inner_dp = data_j_inner['contrast_dt'], data_j_inner['contrast_dp']

        factor = -3*fl_param['H0']**2*om/(2*c.to('km/s').value**2)*(1+redshift_j_inner)*midpoint_j_inner * (midpoint_j - midpoint_j_inner) / midpoint_j

        contrast_dt_map_sum += factor * contrast_j_inner_dt *(interval_j_inner[1]-interval_j_inner[0])
        contrast_dp_map_sum += factor * contrast_j_inner_dp *(interval_j_inner[1]-interval_j_inner[0])
        
    return contrast_dt_map_sum,contrast_dp_map_sum

def ll_sum(fl_param,task_list,cache_dir,j,om,Nside_map):

    print(f'### post-Born lens summation Phi {j} ...')

    _, _, midpoint_j = task_list[j]
    phi_dtt_map_sum = np.zeros(Nside_map**2 * 12)
    phi_dtp_map_sum = np.zeros(Nside_map**2 * 12)
    phi_dpt_map_sum = np.zeros(Nside_map**2 * 12)
    phi_dpp_map_sum = np.zeros(Nside_map**2 * 12)
    for j_inner in range(j):
        file_path_j_inner = os.path.join(cache_dir, f"phi_dd/phi_dd_{j_inner}.npz")
        _, interval_j_inner, midpoint_j_inner = task_list[j_inner]

        if os.path.exists(file_path_j_inner):
            try:
                data_j_inner = np.load(file_path_j_inner)
                phi_j_inner_dtt, phi_j_inner_dpp, phi_j_inner_dtp, phi_j_inner_dpt = data_j_inner['phi_dtt'], data_j_inner['phi_dpp'], data_j_inner['phi_dtp'], data_j_inner['phi_dpt']
            except Exception as e:
                print(f"Error loading file {file_path_j_inner}.")
        else:
            print(f'### Second partial Phi {j_inner} not exist.')
            prepare_phi_dd(fl_param,om,task_list,cache_dir,j_inner)
            data_j_inner = np.load(file_path_j_inner)
            phi_j_inner_dtt, phi_j_inner_dpp, phi_j_inner_dtp, phi_j_inner_dpt = data_j_inner['phi_dtt'], data_j_inner['phi_dpp'], data_j_inner['phi_dtp'], data_j_inner['phi_dpt']

        factor = -2*midpoint_j_inner * (midpoint_j - midpoint_j_inner) / midpoint_j

        phi_dtt_map_sum += factor * phi_j_inner_dtt *(interval_j_inner[1]-interval_j_inner[0])
        phi_dtp_map_sum += factor * phi_j_inner_dtp *(interval_j_inner[1]-interval_j_inner[0])
        phi_dpt_map_sum += factor * phi_j_inner_dpt *(interval_j_inner[1]-interval_j_inner[0])
        phi_dpp_map_sum += factor * phi_j_inner_dpp *(interval_j_inner[1]-interval_j_inner[0])
        
    return phi_dtt_map_sum,phi_dtp_map_sum,phi_dpt_map_sum,phi_dpp_map_sum

def lens_shell_prepare(fl_param, pars=None, random_seed=0, cache_dir=None, mpi=True,new=False):

    print_message('###### Starting process lens shells... ######',mpi)

    if mpi:
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

    if pars is None:
        pars = camb.CAMBparams()
        pars.set_cosmology(H0=fl_param['H0'], ombh2=fl_param['ombh2'], omch2=fl_param['omch2'], 
            tau=fl_param['tau'], mnu=fl_param['mnu'], omk=fl_param['omk'])
        pars.InitPower.set_params(As=fl_param['As'], ns=fl_param['ns'])
    results = camb.get_results(pars)

    Nside_map, Nside_shell, num_shells, n, N, lmax, lmin, lmax_analysis, h, root_local, root_cluster,mixed_Nbody_folder_list,mixed_Nbody_suffix = read_param(fl_param).values()

    if cache_dir is None:
        cache_dir = os.path.join(root_local,f'cache/lensing_generate_{n}Boxshells_{N}Gshells_{mixed_Nbody_suffix}_lensingNside{Nside_map}_lmax{lmax}/seed{random_seed}')
        os.makedirs(cache_dir, exist_ok=True)

    ######################## load shell arrangement data #########
    boxsize_list = np.loadtxt(os.path.join(root_local, f"{mixed_Nbody_folder_list[0]}/boxsize_list.txt"))
    intervals_G = np.load(os.path.join(root_local, f'shells/shell_intervals_{n}Boxshells_{N}Gshell_G.npy'))
    midpoints_G = np.load(os.path.join(root_local, f'shells/shell_eff_midpoints_{n}Boxshells_{N}Gshell_G.npy'))
    redshifts_G = results.redshift_at_comoving_radial_distance(np.asarray(midpoints_G)/h)
    total_G_shells = len(redshifts_G)

    ######################## Gaussian shells ########################
    if mpi:
        for counter in range(rank, total_G_shells, size):
            redshift = redshifts_G[counter]
            midpoint_G = midpoints_G[counter]/h
            interval_G = intervals_G[counter]/h
            prepare_Gaussian_lens_shell(fl_param,cache_dir,pars,redshift,midpoint_G,interval_G,counter,random_seed,new=new)
        comm.Barrier()
        if rank==0:
            print('### Gaussian shells ready')

    else:
        for counter in range(total_G_shells):
            redshift = redshifts_G[counter]
            midpoint_G = midpoints_G[counter]/h
            interval_G = intervals_G[counter]/h
            prepare_Gaussian_lens_shell(fl_param,cache_dir,pars,redshift,midpoint_G,interval_G,counter,random_seed,new=new)
        print('### Gaussian shells ready')

    ######################## Nbody shells ########################
    l_corr_list = []
    for idx in range(1,len(num_shells)):
        midpoints = np.load(os.path.join(root_local, f'shells/shell_eff_midpoints_{n}Boxshells_b{idx+1}.npy'))/h
        l_corr = np.int_((2*np.pi)*midpoints/(boxsize_list[idx]/2/h)-0.5)
        l_corr_list.extend(l_corr)
    min_l_corr = min(l_corr_list)

    task_list = []
    for idx in reversed(range(len(boxsize_list))):

        midpoints = np.load(os.path.join(root_local, f'shells/shell_eff_midpoints_{n}Boxshells_b{idx+1}.npy'))/h
        intervals = np.load(os.path.join(root_local, f'shells/shell_intervals_{n}Boxshells_b{idx+1}.npy'))/h
        redshifts = results.redshift_at_comoving_radial_distance(midpoints)
        np.random.seed(random_seed+idx)
        Nbody_mixed_index_list = np.random.choice(np.arange(len(mixed_Nbody_folder_list)),len(midpoints))
        
        for counter in range(len(midpoints)):
            midpoint = midpoints[counter]
            interval = intervals[counter]
            redshift = redshifts[counter]
            Nbody_mixed_index = Nbody_mixed_index_list[counter]
            task_list.append((midpoint,interval,redshift, idx, counter,Nbody_mixed_index))

    if mpi:
        for task_id in range(rank, len(task_list), size):
            midpoint, interval, redshift, idx, counter,Nbody_mixed_index = task_list[task_id]
            overall_counter = task_id+total_G_shells
            Nbody_folder = mixed_Nbody_folder_list[Nbody_mixed_index]
            prepare_Nbody_lens_shell(fl_param,cache_dir,pars,redshift,midpoint,interval,idx,counter,min_l_corr,task_id,boxsize_list,Nbody_folder,overall_counter,random_seed,new=new)
        comm.Barrier()
        if rank==0:
            print('### Nbody shells ready')
    else:
        for counter in range(len(task_list)):
            midpoint, interval, redshift, idx, counter,Nbody_mixed_index = task_list[task_id]
            overall_counter = task_id+total_G_shells
            Nbody_folder = mixed_Nbody_folder_list[Nbody_mixed_index]
            prepare_Nbody_lens_shell(fl_param,cache_dir,pars,redshift,midpoint,interval,idx,counter,min_l_corr,task_id,boxsize_list,Nbody_folder,overall_counter,random_seed,new=new)
        print('### Nbody shells ready')

    return cache_dir

def kappa_Born(fl_param, pars=None, cache_dir=None, random_seed=0, mpi=True, new=False):

    print_message('###### Getting kappa (Born approximation)... ######',mpi)

    Nside_map, Nside_shell, num_shells, n, N, lmax, lmin, lmax_analysis, h, root_local, root_cluster,mixed_Nbody_folder_list,mixed_Nbody_suffix = read_param(fl_param).values()
    
    lensing_path = os.path.join(root_local, f"lensing/lensing-Nside{Nside_map}_shell-Nside{Nside_shell}_lmax{lmax}")
    os.makedirs(lensing_path, exist_ok=True)

    file_path = os.path.join(lensing_path, f"map_kappa_{n}Boxshells_{N}Gshells_{mixed_Nbody_suffix}_lensingNside{Nside_map}_lmax{lmax}_seed{random_seed}.fits")
    if not os.path.exists(file_path) or new:
        if cache_dir is None:
            cache_dir = os.path.join(root_local,f'cache/lensing_generate_{n}Boxshells_{N}Gshells_{mixed_Nbody_suffix}_lensingNside{Nside_map}_lmax{lmax}/seed{random_seed}')

        contrast_map_path = os.path.join(cache_dir, f'shell_contrast')
        if not os.path.exists(contrast_map_path):
            print_message('Need to generate the lens shell contrast maps first.',mpi)
            print_message('Starting to generate the lens shell contrast maps...',mpi)
            cache_dir = lens_shell_prepare(fl_param, pars=pars, random_seed=random_seed, cache_dir=cache_dir, mpi=mpi,new=False)
        else:
            print_message(f'### Lens shell contrast maps found at {contrast_map_path}. Proceeding kappa Born approximation term...',mpi)
  
        if pars is None:
            pars = camb.CAMBparams()
            pars.set_cosmology(H0=fl_param['H0'], ombh2=fl_param['ombh2'], omch2=fl_param['omch2'], 
                tau=fl_param['tau'], mnu=fl_param['mnu'], omk=fl_param['omk'])
            pars.InitPower.set_params(As=fl_param['As'], ns=fl_param['ns'])

        results = camb.get_results(pars)
        chistar_Mpc = (results.conformal_time(0) - results.tau_maxvis)
        omde = results.get_Omega('de', z=0)
        omc = results.get_Omega('cdm', z=0)
        omb = results.get_Omega('baryon', z=0)
        omrad = results.get_Omega('photon', z=0)
        omneutrino = results.get_Omega('neutrino', z=0)
        omnu = results.get_Omega('nu', z=0)
        om = omc+omb+omrad+omneutrino+omnu

        task_list = []
        boxsize_list = np.loadtxt(os.path.join(root_local, f"{mixed_Nbody_folder_list[0]}/boxsize_list.txt"))
        midpoints_G = np.load(os.path.join(root_local, f'shells/shell_eff_midpoints_{n}Boxshells_{N}Gshell_G.npy'))/h
        intervals_G = np.load(os.path.join(root_local, f'shells/shell_intervals_{n}Boxshells_{N}Gshell_G.npy'))/h
        redshifts_G = results.redshift_at_comoving_radial_distance(np.asarray(midpoints_G))
        for counter in range(len(intervals_G)):
            task_list.append([redshifts_G[counter],intervals_G[counter],midpoints_G[counter]])
        for idx in reversed(range(len(boxsize_list))):
            intervals = np.load(os.path.join(root_local, f'shells/shell_intervals_{n}Boxshells_b{idx+1}.npy'))/h
            midpoints = np.load(os.path.join(root_local, f'shells/shell_eff_midpoints_{n}Boxshells_b{idx+1}.npy'))/h
            redshifts = results.redshift_at_comoving_radial_distance(midpoints)
            for counter in range(len(intervals)):
                task_list.append([redshifts[counter],intervals[counter],midpoints[counter]])

        if mpi:
            comm = MPI.COMM_WORLD
            rank = comm.Get_rank()
            size = comm.Get_size()
            comm.Barrier()
            local_map_sum = np.zeros(Nside_map**2 * 12)
            for shell_idx in range(rank, len(task_list), size):
                [redshift,interval,midpoint] = task_list[shell_idx]
                alm_contrast = np.load(os.path.join(cache_dir, f'shell_contrast/alm_contrast_{n}Boxshells_{N}Gshells_{mixed_Nbody_suffix}_lensingNside{Nside_map}_counter{shell_idx}.npy'))
                map_contrast = hp.sphtfunc.alm2map(alm_contrast, Nside_map, lmax=lmax)
                local_map_sum = lensing_summation(fl_param, map_contrast, local_map_sum, redshift, midpoint, chistar_Mpc, interval, om=om)
            if rank == 0:
                map_sum = np.zeros_like(local_map_sum)
            else:
                map_sum = None
            comm.Reduce(local_map_sum, map_sum, op=MPI.SUM, root=0)
            if rank == 0:
                hp.fitsfunc.write_map(file_path, map_sum,overwrite=True)
                print(f'### kappa map (Born approximation) saved at {file_path}')
        else:
            map_sum = np.zeros(Nside_map**2 * 12)
            for shell_idx in range(len(task_list)):
                [redshift,interval,midpoint] = task_list[shell_idx]
                alm_contrast = np.load(os.path.join(cache_dir, f'shell_contrast/alm_contrast_{n}Boxshells_{N}Gshells_{mixed_Nbody_suffix}_lensingNside{Nside_map}_counter{shell_idx}.npy'))
                map_contrast = hp.sphtfunc.alm2map(alm_contrast, Nside_map, lmax=lmax)
                map_sum = lensing_summation(fl_param, map_contrast, map_sum, redshift, midpoint, chistar_Mpc, interval, om=om)
            hp.fitsfunc.write_map(file_path, map_sum,overwrite=True)
            print(f'### kappa map (Born approximation) saved at {file_path}')
    else:
        print_message(f'### kappa map (Born approximation) already exist at {file_path}',mpi)
        print_message('### set new=True for regenrate',mpi)

def kappa_ll(fl_param, pars=None, cache_dir=None, random_seed=0, mpi=True, new=False):

    print_message('###### Getting kappa post-Born corrction (lens-lens)... ######',mpi)

    Nside_map, Nside_shell, num_shells, n, N, lmax, lmin, lmax_analysis, h, root_local, root_cluster,mixed_Nbody_folder_list,mixed_Nbody_suffix = read_param(fl_param).values()
    
    lensing_path = os.path.join(root_local, f"lensing/lensing-Nside{Nside_map}_shell-Nside{Nside_shell}_lmax{lmax}")
    file_path = os.path.join(lensing_path, f"map_kappa_ll_{n}Boxshells_{N}Gshells_{mixed_Nbody_suffix}_lensingNside{Nside_map}_lmax{lmax}_seed{random_seed}.fits")
    if not os.path.exists(file_path) or new:
        if cache_dir is None:
            cache_dir = os.path.join(root_local,f'cache/lensing_generate_{n}Boxshells_{N}Gshells_{mixed_Nbody_suffix}_lensingNside{Nside_map}_lmax{lmax}/seed{random_seed}')

        contrast_map_path = os.path.join(cache_dir, f'shell_contrast')
        if not os.path.exists(contrast_map_path):
            print_message('Need to generate the lens shell contrast maps first.',mpi)
            print_message('Starting to generate the lens shell contrast maps...',mpi)
            cache_dir = lens_shell_prepare(fl_param, pars=pars, random_seed=random_seed, cache_dir=cache_dir, mpi=mpi,new=new)
        else:
            print_message(f'### Lens shell contrast maps found at {contrast_map_path}. Proceeding kappa lens-lens term...',mpi)
      
        if pars is None:
            pars = camb.CAMBparams()
            pars.set_cosmology(H0=fl_param['H0'], ombh2=fl_param['ombh2'], omch2=fl_param['omch2'], 
                tau=fl_param['tau'], mnu=fl_param['mnu'], omk=fl_param['omk'])
            pars.InitPower.set_params(As=fl_param['As'], ns=fl_param['ns'])

        results = camb.get_results(pars)
        chistar_Mpc = (results.conformal_time(0) - results.tau_maxvis)
        omde = results.get_Omega('de', z=0)
        omc = results.get_Omega('cdm', z=0)
        omb = results.get_Omega('baryon', z=0)
        omrad = results.get_Omega('photon', z=0)
        omneutrino = results.get_Omega('neutrino', z=0)
        omnu = results.get_Omega('nu', z=0)
        om = omc+omb+omrad+omneutrino+omnu

        task_list = []
        boxsize_list = np.loadtxt(os.path.join(root_local, f"{mixed_Nbody_folder_list[0]}/boxsize_list.txt"))
        midpoints_G = np.load(os.path.join(root_local, f'shells/shell_eff_midpoints_{n}Boxshells_{N}Gshell_G.npy'))/h
        intervals_G = np.load(os.path.join(root_local, f'shells/shell_intervals_{n}Boxshells_{N}Gshell_G.npy'))/h
        redshifts_G = results.redshift_at_comoving_radial_distance(np.asarray(midpoints_G))
        for counter in range(len(intervals_G)):
            task_list.append([redshifts_G[counter],intervals_G[counter],midpoints_G[counter]])
        for idx in reversed(range(len(boxsize_list))):
            intervals = np.load(os.path.join(root_local, f'shells/shell_intervals_{n}Boxshells_b{idx+1}.npy'))/h
            midpoints = np.load(os.path.join(root_local, f'shells/shell_eff_midpoints_{n}Boxshells_b{idx+1}.npy'))/h
            redshifts = results.redshift_at_comoving_radial_distance(midpoints)
            for counter in range(len(intervals)):
                task_list.append([redshifts[counter],intervals[counter],midpoints[counter]])

        if mpi:
            comm = MPI.COMM_WORLD
            rank = comm.Get_rank()
            size = comm.Get_size()
            if rank==0:
                os.makedirs(os.path.join(cache_dir,f'phi_dd'), exist_ok=True)
            comm.Barrier()
            for j in range(rank, len(task_list), size):
                prepare_phi_dd(fl_param,om,task_list,cache_dir,j,new=new)
            comm.Barrier()


            local_ll_map_sum = np.zeros(Nside_map**2 * 12)
            for j in range(rank, len(task_list), size):
                _, interval_j, midpoint_j = task_list[j]
                phi_dtt_map_sum,phi_dtp_map_sum,phi_dpt_map_sum,phi_dpp_map_sum = ll_sum(fl_param,task_list,cache_dir,j,om,Nside_map)

                file_path_j = os.path.join(cache_dir, f"phi_dd/phi_dd_{j}.npz")

                data_j = np.load(file_path_j)
                phi_j_dtt, phi_j_dpp, phi_j_dtp, phi_j_dpt = data_j['phi_dtt'], data_j['phi_dpp'], data_j['phi_dtp'], data_j['phi_dpt']

                factor = midpoint_j * (chistar_Mpc - midpoint_j) / chistar_Mpc*(interval_j[1]-interval_j[0])**0.5
                local_ll_map_sum += factor * (phi_j_dtt*phi_dtt_map_sum + phi_j_dtp*phi_dtp_map_sum + phi_j_dpt*phi_dpt_map_sum + phi_j_dpp*phi_dpp_map_sum)
                
            comm.Barrier()

            kappa_ll_map = np.zeros_like(local_ll_map_sum)
            comm.Reduce(local_ll_map_sum, kappa_ll_map, op=MPI.SUM, root=0)

            if rank == 0:
                hp.fitsfunc.write_map(file_path, kappa_ll_map, overwrite=True)
                print(f'### kappa map (lens-lens) saved at {file_path}')

        else:
            os.makedirs(os.path.join(cache_dir,f'phi_dd'), exist_ok=True)
            for j in range(len(task_list)):
                prepare_phi_dd(fl_param,om,task_list,cache_dir,j,new=new)

            kappa_ll_map = np.zeros(Nside_map**2 * 12)
            for j in range(len(task_list)):
                _, interval_j, midpoint_j = task_list[j]
                phi_dtt_map_sum,phi_dtp_map_sum,phi_dpt_map_sum,phi_dpp_map_sum = ll_sum(fl_param,task_list,cache_dir,j,om,Nside_map)
                file_path_j = os.path.join(cache_dir, f"phi_dd/phi_dd_{j}.npz")

                data_j = np.load(file_path_j)
                phi_j_dtt, phi_j_dpp, phi_j_dtp, phi_j_dpt = data_j['phi_dtt'], data_j['phi_dpp'], data_j['phi_dtp'], data_j['phi_dpt']

                factor = midpoint_j * (chistar_Mpc - midpoint_j) / chistar_Mpc*(interval_j[1]-interval_j[0])**0.5
                kappa_ll_map += factor * (phi_j_dtt*phi_dtt_map_sum + phi_j_dtp*phi_dtp_map_sum + phi_j_dpt*phi_dpt_map_sum + phi_j_dpp*phi_dpp_map_sum)
                
            hp.fitsfunc.write_map(file_path, kappa_ll_map, overwrite=True)
            print(f'### kappa map (lens-lens) saved at {file_path}')

    else:
        print_message(f'kappa map (lens-lens) already exist at {file_path}',mpi)
        print_message('set new=True for regenrate',mpi)

def kappa_geo(fl_param, pars=None, cache_dir=None, random_seed=0, mpi=True, new=False):

    print_message('###### Getting kappa post-Born corrction (geodesic)... ######',mpi)

    Nside_map, Nside_shell, num_shells, n, N, lmax, lmin, lmax_analysis, h, root_local, root_cluster,mixed_Nbody_folder_list,mixed_Nbody_suffix = read_param(fl_param).values()
    
    lensing_path = os.path.join(root_local, f"lensing/lensing-Nside{Nside_map}_shell-Nside{Nside_shell}_lmax{lmax}")
    file_path = os.path.join(lensing_path, f"map_kappa_geo_{n}Boxshells_{N}Gshells_{mixed_Nbody_suffix}_lensingNside{Nside_map}_lmax{lmax}_seed{random_seed}.fits")
    if not os.path.exists(file_path) or new:
        if cache_dir is None:
            cache_dir = os.path.join(root_local,f'cache/lensing_generate_{n}Boxshells_{N}Gshells_{mixed_Nbody_suffix}_lensingNside{Nside_map}_lmax{lmax}/seed{random_seed}')

        contrast_map_path = os.path.join(cache_dir, f'shell_contrast')
        if not os.path.exists(contrast_map_path):
            print_message('Need to generate the lens shell contrast maps first.',mpi)
            print_message('Starting to generate the lens shell contrast maps...',mpi)
            cache_dir = lens_shell_prepare(fl_param, pars=pars, random_seed=random_seed, cache_dir=cache_dir, mpi=mpi,new=new)
        else:
            print_message(f'### Lens shell contrast maps found at {contrast_map_path}. Proceeding kappa lens-lens term...',mpi)
      
        if pars is None:
            pars = camb.CAMBparams()
            pars.set_cosmology(H0=fl_param['H0'], ombh2=fl_param['ombh2'], omch2=fl_param['omch2'], 
                tau=fl_param['tau'], mnu=fl_param['mnu'], omk=fl_param['omk'])
            pars.InitPower.set_params(As=fl_param['As'], ns=fl_param['ns'])

        results = camb.get_results(pars)
        chistar_Mpc = (results.conformal_time(0) - results.tau_maxvis)
        omde = results.get_Omega('de', z=0)
        omc = results.get_Omega('cdm', z=0)
        omb = results.get_Omega('baryon', z=0)
        omrad = results.get_Omega('photon', z=0)
        omneutrino = results.get_Omega('neutrino', z=0)
        omnu = results.get_Omega('nu', z=0)
        om = omc+omb+omrad+omneutrino+omnu

        task_list = []
        boxsize_list = np.loadtxt(os.path.join(root_local, f"{mixed_Nbody_folder_list[0]}/boxsize_list.txt"))
        midpoints_G = np.load(os.path.join(root_local, f'shells/shell_eff_midpoints_{n}Boxshells_{N}Gshell_G.npy'))/h
        intervals_G = np.load(os.path.join(root_local, f'shells/shell_intervals_{n}Boxshells_{N}Gshell_G.npy'))/h
        redshifts_G = results.redshift_at_comoving_radial_distance(np.asarray(midpoints_G))
        for counter in range(len(intervals_G)):
            task_list.append([redshifts_G[counter],intervals_G[counter],midpoints_G[counter]])
        for idx in reversed(range(len(boxsize_list))):
            intervals = np.load(os.path.join(root_local, f'shells/shell_intervals_{n}Boxshells_b{idx+1}.npy'))/h
            midpoints = np.load(os.path.join(root_local, f'shells/shell_eff_midpoints_{n}Boxshells_b{idx+1}.npy'))/h
            redshifts = results.redshift_at_comoving_radial_distance(midpoints)
            for counter in range(len(intervals)):
                task_list.append([redshifts[counter],intervals[counter],midpoints[counter]])

        if mpi:
            comm = MPI.COMM_WORLD
            rank = comm.Get_rank()
            size = comm.Get_size()
            if rank==0:
                os.makedirs(os.path.join(cache_dir,f'phi_d'), exist_ok=True)

            comm.Barrier()
            for j in range(rank, len(task_list), size):
                prepare_phi_d(fl_param,om,task_list,cache_dir,j,new=new)
            comm.Barrier()

            local_geo_map_sum = np.zeros(Nside_map**2 * 12)
            for j in range(rank, len(task_list), size):
                _, interval_j, midpoint_j = task_list[j]
                contrast_dt_map_sum,contrast_dp_map_sum = geo_sum(fl_param,task_list,cache_dir,j,om,Nside_map)

                file_path_j = os.path.join(cache_dir, f"phi_d/phi_d_{j}.npz")

                data_j = np.load(file_path_j)
                phi_j_dt, phi_j_dp= data_j['phi_dt'], data_j['phi_dp']

                factor = midpoint_j * (chistar_Mpc - midpoint_j) / chistar_Mpc*(interval_j[1]-interval_j[0])**0.5
                local_geo_map_sum += factor * (phi_j_dt*contrast_dt_map_sum + phi_j_dp*contrast_dp_map_sum)
                
            comm.Barrier()
            kappa_geo_map = np.zeros_like(local_geo_map_sum)
            comm.Reduce(local_geo_map_sum, kappa_geo_map, op=MPI.SUM, root=0)

            if rank == 0:
                hp.fitsfunc.write_map(file_path, kappa_geo_map, overwrite=True)
                print(f'### kappa map (geodesic) saved at {file_path}')
        else:
            os.makedirs(os.path.join(cache_dir,f'phi_d'), exist_ok=True)
            for j in range(len(task_list)):
                prepare_phi_d(fl_param,om,task_list,cache_dir,j,new=new)

            kappa_geo_map = np.zeros(Nside_map**2 * 12)
            for j in range(len(task_list)):
                _, interval_j, midpoint_j = task_list[j]
                contrast_dt_map_sum,contrast_dp_map_sum = geo_sum(fl_param,task_list,cache_dir,j,om,Nside_map)

                file_path_j = os.path.join(cache_dir, f"phi_d/phi_d_{j}.npz")

                data_j = np.load(file_path_j)
                phi_j_dt, phi_j_dp= data_j['phi_dt'], data_j['phi_dp']

                factor = midpoint_j * (chistar_Mpc - midpoint_j) / chistar_Mpc*(interval_j[1]-interval_j[0])**0.5
                kappa_geo_map += factor * (phi_j_dt*contrast_dt_map_sum + phi_j_dp*contrast_dp_map_sum)
                
            hp.fitsfunc.write_map(file_path, kappa_geo_map, overwrite=True)
            print(f'### kappa map (geodesic) saved at {file_path}')

    else:
        print_message(f'kappa map (geodesic) already exist at {file_path}',mpi)
        print_message('set new=True for regenrate',mpi)

