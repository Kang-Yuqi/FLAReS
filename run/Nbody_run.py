import QingLongTool as ql
import matplotlib
import matplotlib.pyplot as plt
import os 
import numpy as np

import configparser
config = configparser.ConfigParser()
config.read('run_config.ini')
root_unify = config['ROOT_PATH']['root_unify']
root_local_base = config['ROOT_PATH']['root_local_base']
root_cluster_base = config['ROOT_PATH']['root_cluster_base']

root_local = root_local_base + root_unify
root_cluster = root_cluster_base + root_unify

os.makedirs(os.path.join(root_local), exist_ok=True)

######################## load param ########################
param_path = os.path.join(root_local, 'ql_param.ini')
ql_param = ql.read_config(param_path)

# ######################## initial condition power spectrum ########################
os.makedirs(os.path.join(root_local, 'nbody_ic'), exist_ok=True)

pk_camb,pk_ngenic = ql.nbody.initial_power_spectrum.Pk(0,nstep=800,kmax=1e2,h_unit=True, k_hunit=True,nonlinear=False,
	H0=ql_param['H0'], ombh2=ql_param['ombh2'], omch2=ql_param['omch2'], 
	tau=ql_param['tau'], mnu=ql_param['mnu'], omk=ql_param['omk'], As=ql_param['As'], 
	ns=ql_param['ns'])

np.savetxt(os.path.join(root_local, 'nbody_ic/ngenic_matterpower.txt'),pk_ngenic)

plt.loglog(pk_camb[:,0],pk_camb[:,1])
plt.ylabel(r'$\mathrm{P}(k)~[(\mathrm{Mpc}/h)^3]$')
plt.xlabel(r'$k[h/\mathrm{Mpc}]$')
plt.title(r'CAMB ($z = 0$)')
plt.savefig(os.path.join(root_local, 'nbody_ic/camb_matterpower'))
plt.close()

plt.plot(pk_ngenic[:,0],pk_ngenic[:,1])
plt.ylabel(r'$\log _{10}([4\pi k^3 \mathrm{P}(k)]^2)~[(\mathrm{Mpc}/h)^3]$')
plt.xlabel(r'$\log _{10}(k)[h/\mathrm{Mpc}]$')
plt.title(r'N-GenIC input ($z = 0$)')
plt.savefig(os.path.join(root_local, 'nbody_ic/ngenic_matterpower'))
plt.close()


######################## Gadget4 param ########################
os.makedirs(os.path.join(root_local, 'nbody_sim'), exist_ok=True)

import camb
pars = camb.CAMBparams()
pars.set_cosmology(H0=ql_param['H0'], ombh2=ql_param['ombh2'], omch2=ql_param['omch2'], 
	tau=ql_param['tau'], mnu=ql_param['mnu'], omk=ql_param['omk'])
pars.InitPower.set_params(As=ql_param['As'], ns=ql_param['ns'])
pars.set_matter_power(redshifts=[0.], kmax=10.0)
results = camb.get_results(pars)
omde = results.get_Omega('de', z=0)
omc = results.get_Omega('cdm', z=0)
omb = results.get_Omega('baryon', z=0)
omrad = results.get_Omega('photon', z=0)
omneutrino = results.get_Omega('neutrino', z=0)
omnu = results.get_Omega('nu', z=0)
om = omc+omb+omrad+omneutrino+omnu
sigma8 = results.get_sigma8()


##### shell arrange ####
os.makedirs(os.path.join(root_local, 'shells'), exist_ok=True)

boxsize_list = []
comoving_shell_end_list = []
for i in reversed(range(int(ql_param['num_box']))):
    boxsize = ql_param['boxsize_max']/(ql_param['boxsize_ratio']**i)
    angle = 180/ql_param['l_eff']
    circum = boxsize/angle*360
    comoving_shell_end = circum/(2*np.pi)
    redshift = results.redshift_at_comoving_radial_distance(comoving_shell_end/(ql_param['H0']/100))
    boxsize_list.append(boxsize)
    comoving_shell_end_list.append(comoving_shell_end)
    print(boxsize,redshift,comoving_shell_end/(ql_param['H0']/100))

grouped_intervals = []
grouped_midpoints = []
for i in range(int(ql_param['num_box'])):
    if i == 0:
        split = np.linspace(0,comoving_shell_end_list[i],2*ql_param['num_shells'][i]+1)
        midpoint = split[1::2]
        intervals = np.column_stack([split[::2][:-1], split[::2][1:]])
    else:
        split = np.linspace(comoving_shell_end_list[i-1],comoving_shell_end_list[i],2*ql_param['num_shells'][i]+1)
        midpoint = split[1::2]
        intervals = np.column_stack([split[::2][:-1], split[::2][1:]])
    grouped_midpoints.append(midpoint)
    grouped_intervals.append(intervals)

n_splits=len(boxsize_list)

np.savetxt(f"{os.path.join(root_local, f'nbody_sim/boxsize_list.txt')}",boxsize_list, fmt=('%.2f'))
for idx in range(n_splits):
    print(grouped_midpoints[idx][::-1],results.redshift_at_comoving_radial_distance(grouped_midpoints[idx][::-1]/(ql_param['H0']/100)))
    os.makedirs(os.path.join(root_local, f'nbody_sim/nbody{idx+1}'), exist_ok=True)
    intervals_time = [1/(1+results.redshift_at_comoving_radial_distance(interval/(ql_param['H0']/100)))[::-1] for interval in grouped_intervals[idx]][::-1]
    intervals_with_indices = [[int(i),int(0), time[0], time[1]] for i, time in enumerate(intervals_time)]
    outputs_step_in_time = 1/(1+results.redshift_at_comoving_radial_distance(grouped_midpoints[idx][::-1]/(ql_param['H0']/100)))
    np.savetxt(f"{os.path.join(root_local, f'nbody_sim/lightcone_b{idx+1}.txt')}",intervals_with_indices, fmt=('%d', '%d', '%.6f', '%.6f'))
    np.savetxt(f"{os.path.join(root_local, f'nbody_sim/outputs_b{idx+1}.txt')}",outputs_step_in_time, fmt=('%.6f'))
    
    np.save(f"{os.path.join(root_local, f'shells/shell_intervals_b{idx+1}')}",grouped_intervals[idx][::-1])
    np.save(f"{os.path.join(root_local, f'shells/shell_midpoints_b{idx+1}')}",grouped_midpoints[idx][::-1])

    z_start = int(ql_param['z_start'])
    TimeBegin = "%.6f" %(1/(1+z_start))
    TimeMax = "%.6f" %outputs_step_in_time[-1]
    BoxSize = boxsize_list[idx]
    shell_thickness = grouped_intervals[idx][0][1]-grouped_intervals[idx][0][0]

    print(grouped_intervals[idx][0][1],shell_thickness)
    changes = {
        "OutputDir": f"{os.path.join(root_cluster, f'nbody_sim/nbody{idx+1}')}",
        "OutputListFilename": f"{os.path.join(root_cluster, f'nbody_sim/outputs_b{idx+1}.txt')}",
        "TimeLimitCPU": f"{ql_param['TimeLimitCPU']}",
        "CpuTimeBetRestartFile": f"{ql_param['CpuTimeBetRestartFile']}",
        "MaxMemSize": f"{ql_param['MaxMemSize']}",
        "TimeBegin": f"{TimeBegin}",
        "TimeMax": f"{TimeMax}",
        "Omega0": f"{om}",
        "OmegaLambda": f"{omde}",
        "OmegaBaryon": f"{omb}",
        "HubbleParam": f"{ql_param['H0']/100}",
        "BoxSize": f"{BoxSize}",
        "NumFilesPerSnapshot": f"{ql_param['NumFilesPerSnapshot']}",
        "NSample": f"{ql_param['NSample']}",
        "GridSize": f"{ql_param['GridSize']}",
        "Sigma8": f"{sigma8[0]}",
        "PowerSpectrumFile": f"{os.path.join(root_cluster, f'nbody_ic/ngenic_matterpower.txt')}",
    }

    ql.modify_txt_file(os.path.join(root_local_base, 'profile_sample/param.txt'), os.path.join(root_local, f'nbody_sim/param_{idx+1}.txt'), changes)


######################## Gadget4 qsub ########################
os.makedirs(os.path.join(root_local, 'katana'), exist_ok=True)

core_num = (int(ql_param['mpi_cpu_num'])-1)//20+1

if int(ql_param['mpi_cpu_num'])>20:
    cpu_num = 20
else:
    cpu_num = int(ql_param['mpi_cpu_num'])


if int(ql_param['mpi_cpu_num'])<=20:
    mem = int(int(ql_param['MaxMemSize'])*int(ql_param['mpi_cpu_num'])/1024)+2*int(ql_param['mpi_cpu_num'])
else:
    mem = int(int(ql_param['MaxMemSize'])*20/1024)+2*20

import time
def seconds_to_time(seconds):
    hours, remainder = divmod(seconds, 3600)
    time_struct = time.gmtime(remainder)
    return f"{int(hours):02}:{time_struct.tm_min:02}:{time_struct.tm_sec:02}"

wall_time = seconds_to_time(int(ql_param['TimeLimitCPU']))

for idx in range(n_splits):
    lines = [
        "#!/bin/bash\n",
        f"#PBS -N g4_{idx+1}\n",
        f"#PBS -l select={core_num}:ncpus={cpu_num}:mpiprocs={cpu_num}:mem={mem}gb\n",
        f"#PBS -l walltime={wall_time}\n",
        "#PBS -l ib=True\n",
        "#PBS -j oe\n",
        "#PBS -m ae\n",
        "\n",
        "cd\n",
        ". module_list_Gadget4\n",
        "cd /gadget4\n",
        f"mpirun -np {cpu_num*core_num} ./Gadget4 {root_cluster}nbody_sim/param_{idx+1}.txt\n"
    ]

    file_path = os.path.join(root_local, f'cluster/qsub_nbody_{idx+1}.pbs')

    with open(file_path, 'w') as file:
        file.writelines(lines)



        