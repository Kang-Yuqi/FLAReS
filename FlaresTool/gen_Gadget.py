import numpy as np
import os
import pkg_resources
import time
import camb 

def seconds_to_time(seconds):
    hours, remainder = divmod(seconds, 3600)
    time_struct = time.gmtime(remainder)
    return f"{int(hours):02}:{time_struct.tm_min:02}:{time_struct.tm_sec:02}"

def get_project_suffix(root_path):
    base_name = os.path.basename(os.path.normpath(root_path))
    
    if base_name.startswith('gadget4_'):
        target_string = base_name[len('gadget4_'):]
        return target_string
    else:
        return base_name

import re
import pkg_resources

def new_gadget_file(output_file_path, changes):

    resource_path = 'data/Gadget_param.txt'
    input_file_content = pkg_resources.resource_string('QingLongTool', resource_path).decode('utf-8')
    
    lines = input_file_content.split('\n')
    
    for idx, line in enumerate(lines):
        for key, value in changes.items():
            if line.startswith(key):
                match = re.search(f"{key}(\s+)", line)
                if match:
                    spaces = match.group(1)
                    prefix = line.split(key)[0] + key + spaces
                    lines[idx] = prefix + value

    with open(output_file_path, 'w', encoding='utf-8') as file:
        file.write('\n'.join(lines))

def gadget_ini_with_arrange_shells(fl_param, root_local, root_cluster):

    num_shells = fl_param['num_shells']
    N = fl_param['Gaussian_shell_steps']
    n = sum(num_shells)
    h = fl_param['H0']/100

    ######################## cosmology with CAMB ##################
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=fl_param['H0'], ombh2=fl_param['ombh2'], omch2=fl_param['omch2'], 
        tau=fl_param['tau'], mnu=fl_param['mnu'], omk=fl_param['omk'])
    pars.InitPower.set_params(As=fl_param['As'], ns=fl_param['ns'])
    pars.set_matter_power(redshifts=[0.], kmax=10.0)
    results = camb.get_results(pars)
    chistar = (results.conformal_time(0) - results.tau_maxvis)*fl_param['H0']/100 # the comoving distance of CMB

    print('The comoving distance of last scattering surface', chistar/h)
    print('The redshift of last scattering surface', results.redshift_at_comoving_radial_distance(chistar/h))

    omde = results.get_Omega('de', z=0)
    omc = results.get_Omega('cdm', z=0)
    omb = results.get_Omega('baryon', z=0)
    omrad = results.get_Omega('photon', z=0)
    omneutrino = results.get_Omega('neutrino', z=0)
    omnu = results.get_Omega('nu', z=0)
    om = omc+omb+omrad+omneutrino+omnu
    sigma8 = results.get_sigma8()


    if fl_param['shell_arrange_exist']==1:
        shell_save_root = fl_param['shell_arrange_from_folder']
        print(f"shell arrangement loaded from {fl_param['shell_arrange_from_folder']}")
        
        boxsize_list = np.loadtxt(os.path.join(shell_save_root, f"{fl_param['Nbody_folder']}/boxsize_list.txt"))
        np.savetxt(os.path.join(root_local, f"{fl_param['Nbody_folder']}/boxsize_list.txt"), boxsize_list)

        # Nbody shells
        for idx in range(int(fl_param['num_box'])):
            os.makedirs(os.path.join(root_local, f"{fl_param['Nbody_folder']}/nbody{idx+1}"), exist_ok=True)
            intervals_dis_ratio = np.load(os.path.join(shell_save_root, f'shells/shell_intervals_dis_ratio_{n}Boxshells_b{idx+1}.npy')) 
            eff_midpoints_dis_ratio = np.load(os.path.join(shell_save_root, f'shells/shell_eff_midpoints_dis_ratio_{n}Boxshells_b{idx+1}.npy'))
            intervals = intervals_dis_ratio * chistar
            eff_midpoints = eff_midpoints_dis_ratio * chistar
            np.save(os.path.join(root_local, f'shells/shell_intervals_{n}Boxshells_b{idx+1}'),intervals) 
            np.save(os.path.join(root_local, f'shells/shell_eff_midpoints_{n}Boxshells_b{idx+1}'),eff_midpoints)

        # Gaussian shells 
        eff_midpoints_G_dis_ratio = np.load(os.path.join(shell_save_root, f'shells/shell_eff_midpoints_dis_ratio_{n}Boxshells_{N}Gshell_G.npy')) 
        intervals_G_dis_ratio = np.load(os.path.join(shell_save_root, f'shells/shell_intervals_dis_ratio_{n}Boxshells_{N}Gshell_G.npy')) 
        eff_midpoints_G = eff_midpoints_G_dis_ratio*chistar
        intervals_G = intervals_G_dis_ratio*chistar
        np.save(os.path.join(root_local, f'shells/shell_eff_midpoints_{n}Boxshells_{N}Gshell_G'),eff_midpoints_G)
        np.save(os.path.join(root_local, f'shells/shell_intervals_{n}Boxshells_{N}Gshell_G'),intervals_G)

    else:
        print('building new shell arrangement')
        # comoving distance of change box-size 
        boxsize_list = []
        comoving_shell_end_list = []
        redshift_end_list = []
        for i in reversed(range(int(fl_param['num_box']))):
            boxsize = fl_param['boxsize_max']/(fl_param['boxsize_ratio']**i)
            comoving_shell_end = (fl_param['l_eff']+0.5)/(2*np.pi)*boxsize
            redshift_end = results.redshift_at_comoving_radial_distance(comoving_shell_end/h)
            boxsize_list.append(boxsize)
            comoving_shell_end_list.append(comoving_shell_end)
            redshift_end_list.append(redshift_end)
        np.savetxt(os.path.join(root_local, f"{fl_param['Nbody_folder']}/boxsize_list.txt"), boxsize_list, fmt='%.2f')

        # Nbody shells 
        for idx in range(int(fl_param['num_box'])):
            os.makedirs(os.path.join(root_local, f"{fl_param['Nbody_folder']}/nbody{idx+1}"), exist_ok=True)
            # if idx == 0: # smallest box, include all space from the smallest comoving_shell_end to observer
            #     split = np.linspace(0,comoving_shell_end_list[idx],2*fl_param['num_shells'][idx]+1)
            #     midpoints = split[1::2][::-1]
            #     intervals = np.column_stack([split[::2][:-1], split[::2][1:]])[::-1]
            # else:
            #     split = np.linspace(comoving_shell_end_list[idx-1],comoving_shell_end_list[idx],2*fl_param['num_shells'][idx]+1)
            #     midpoints = split[1::2][::-1]
            #     intervals = np.column_stack([split[::2][:-1], split[::2][1:]])[::-1]

            if idx == 0:  # smallest box, include all space from the smallest comoving_shell_end to observer
                split = np.linspace(0, comoving_shell_end_list[idx], fl_param['num_shells'][idx] + 1)
                intervals = np.column_stack([split[:-1], split[1:]])[::-1]
                eff_midpoints = 3/4*(intervals[:,1]**4-intervals[:,0]**4)/(intervals[:,1]**3-intervals[:,0]**3)
            else:
                split = np.linspace(comoving_shell_end_list[idx - 1], comoving_shell_end_list[idx], fl_param['num_shells'][idx] + 1)
                intervals = np.column_stack([split[:-1], split[1:]])[::-1]
                eff_midpoints = 3/4*(intervals[:,1]**4-intervals[:,0]**4)/(intervals[:,1]**3-intervals[:,0]**3)

            # geometry
            eff_midpoints_dis_ratio = eff_midpoints/chistar
            intervals_dis_ratio = intervals/chistar
            with open(os.path.join(root_local,f"{fl_param['Nbody_folder']}/Nbody_arrangement.txt"), 'a') as file:
                file.write(f'boxsize={boxsize_list[idx]} Mpc/h \n')
                file.write(f'the farthest shell edged at z={redshift_end_list[idx]}, chi={comoving_shell_end_list[idx]} Mpc/h\n')
                file.write(f'shell-thickness={intervals[-1][1] - intervals[-1][0]} Mpc/h\n\n')

            print('boxsize=',boxsize_list[idx],'Mpc/h \n','the farthest shell edged at z =',redshift_end_list[idx],', chi =',comoving_shell_end_list[idx],'Mpc/h')
            print(' shell-thickness =',intervals[-1][1]-intervals[-1][0],'Mpc/h')

            np.save(os.path.join(root_local, f'shells/shell_intervals_{n}Boxshells_b{idx+1}'),intervals) 
            np.save(os.path.join(root_local, f'shells/shell_eff_midpoints_{n}Boxshells_b{idx+1}'),eff_midpoints)
            np.save(os.path.join(root_local, f'shells/shell_intervals_dis_ratio_{n}Boxshells_b{idx+1}'),intervals_dis_ratio) 
            np.save(os.path.join(root_local, f'shells/shell_eff_midpoints_dis_ratio_{n}Boxshells_b{idx+1}'),eff_midpoints_dis_ratio)

        # Gaussian shells 
        split = np.linspace(intervals[0][1],chistar, N+1)
        intervals_G = np.column_stack([split[:-1], split[1:]])[::-1]
        eff_midpoints_G = np.sqrt((intervals_G[:,1]**3-intervals_G[i:,0]**3)/(intervals_G[:,1]-intervals_G[i:,0])/3)
        # geometry
        eff_midpoints_G_dis_ratio = eff_midpoints_G/chistar
        intervals_G_dis_ratio = intervals_G/chistar
        np.save(os.path.join(root_local, f'shells/shell_eff_midpoints_{n}Boxshells_{N}Gshell_G'),eff_midpoints_G)
        np.save(os.path.join(root_local, f'shells/shell_intervals_{n}Boxshells_{N}Gshell_G'),intervals_G)
        np.save(os.path.join(root_local, f'shells/shell_eff_midpoints_dis_ratio_{n}Boxshells_{N}Gshell_G'),eff_midpoints_G_dis_ratio)
        np.save(os.path.join(root_local, f'shells/shell_intervals_dis_ratio_{n}Boxshells_{N}Gshell_G'),intervals_G_dis_ratio)


    # Gadget setting
    for idx in range(int(fl_param['num_box'])):
        midpoints = np.load(os.path.join(root_local, f'shells/shell_eff_midpoints_{n}Boxshells_b{idx+1}.npy'))
        redshifts = results.redshift_at_comoving_radial_distance(midpoints/h)
        outputs_step_in_time = 1/(1+redshifts)
        np.savetxt(os.path.join(root_local, f"{fl_param['Nbody_folder']}/outputs_{n}Boxshells_b{idx+1}.txt"), outputs_step_in_time, fmt=('%.10f'))

        z_start = int(fl_param['z_start'])
        TimeBegin = "%.6f" %(1/(1+z_start))
        TimeMax = "%.6f" %outputs_step_in_time[-1]
        BoxSize = boxsize_list[idx]

        # Softening length
        particle_mean_interval = BoxSize/fl_param['NSample']
        SofteningComoving = np.round(particle_mean_interval * fl_param['Softening_factor'],3)
        SofteningMaxPhys = SofteningComoving*0.5

        changes = {
            "OutputDir": os.path.join(root_cluster, f"{fl_param['Nbody_folder']}/nbody{idx+1}"),
            "OutputListFilename": os.path.join(root_cluster, f"{fl_param['Nbody_folder']}/outputs_{n}Boxshells_b{idx+1}.txt"),
            "TimeLimitCPU": f"{fl_param['TimeLimitCPU']}",
            "CpuTimeBetRestartFile": f"{fl_param['CpuTimeBetRestartFile']}",
            "MaxMemSize": f"{fl_param['MaxMemSize']}",
            "TimeBegin": f"{TimeBegin}",
            "TimeMax": f"{TimeMax}",
            "Omega0": f"{om}",
            "OmegaLambda": f"{omde}",
            "OmegaBaryon": f"{omb}",
            "HubbleParam": f"{fl_param['H0']/100}",
            "BoxSize": f"{BoxSize}",
            "NumFilesPerSnapshot": f"{fl_param['NumFilesPerSnapshot']}",
            "SofteningComovingClass0" : f"{SofteningComoving}",
            "SofteningMaxPhysClass0" : f"{SofteningMaxPhys}",
            "NSample": f"{fl_param['NSample']}",
            "GridSize": f"{fl_param['GridSize']}",
            "Seed": f"{fl_param['rand_seed']*(idx+1)}",
            "Sigma8": f"{sigma8[0]}",
            "PowerSpectrumFile": f"{os.path.join(root_cluster, f'nbody_ic/ngenic_matterpower.txt')}",
        }

        new_gadget_file( os.path.join(root_local, f"{fl_param['Nbody_folder']}/param_{idx+1}.txt"), changes)


def cluster_qsub(fl_param, root_local, root_cluster,gadget_path= None ,qsub_output_path=None):
    suffix = get_project_suffix(root_cluster)

    os.makedirs(os.path.join(root_local, 'cluster'), exist_ok=True)

    core_num = (int(fl_param['mpi_cpu_num'])-1)//4+1

    if int(fl_param['mpi_cpu_num'])>4:
        cpu_num = 4
    else:
        cpu_num = int(fl_param['mpi_cpu_num'])


    if int(fl_param['mpi_cpu_num'])<=4:
        mem = int(int(fl_param['MaxMemSize']+500)*int(fl_param['mpi_cpu_num'])/1024)+2*int(fl_param['mpi_cpu_num'])
    else:
        mem = int(int(fl_param['MaxMemSize']+500)*4/1024)+2*4

    wall_time = seconds_to_time(int(fl_param['TimeLimitCPU']))

    for idx in range(int(fl_param['num_box'])):
        lines = [
            "#!/bin/bash\n",
            f"#PBS -N {suffix}_{idx+1}_{fl_param['Nbody_folder']}\n",
            f"#PBS -l select={core_num}:ncpus={cpu_num}:mpiprocs={cpu_num}:tmpmpi=true:mem={mem}gb\n",
            f"#PBS -l walltime={wall_time}\n",
            "#PBS -l ib=True\n",
            "#PBS -j oe\n",
            f"#PBS -o {qsub_output_path}\n",
            "#PBS -m ae\n",
            "\n",
            "cd\n",
            ". module_list_Gadget4\n",
            f"cd {gadget_path}\n",
            f"mpirun -np {cpu_num*core_num} ./Gadget4 {root_cluster}{fl_param['Nbody_folder']}/param_{idx+1}.txt\n"
        ]

        file_path = os.path.join(root_local, f"cluster/qsub_nbody_{idx+1}_{fl_param['Nbody_folder']}.pbs")

        with open(file_path, 'w') as file:
            file.writelines(lines)

