import FlaresTool as fl
import matplotlib.pyplot as plt
import os 
import numpy as np
######################## load param #########################
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-p', '--param-file', default='../param/param_val_sim1.ini', type=str, help='Path to the parameter file')
args = parser.parse_args()
fl_param = fl.read_config(args.param_file)

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
os.makedirs(os.path.join(root_local), exist_ok=True)
os.makedirs(os.path.join(root_local, fl_param['Nbody_folder']), exist_ok=True)
os.makedirs(os.path.join(root_local, 'shells'), exist_ok=True)
os.makedirs(os.path.join(root_local, 'nbody_ic'), exist_ok=True)

######################## initial condition power spectrum ########################
pk_camb,pk_ngenic = fl.initial_power_spectrum.Pk(0,fl_param,nstep=1000,kmax=1e2,h_unit=True, k_hunit=True,nonlinear=False)
np.savetxt(os.path.join(root_local, 'nbody_ic/ngenic_matterpower.txt'),pk_ngenic)

# plot initial power spectrum 
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

######################## generate Gadget4 file and file for cluster run ########################
fl.gen_Gadget.gadget_ini_with_arrange_shells(fl_param, root_local, root_cluster)
fl.gen_Gadget.cluster_qsub(fl_param, root_local, root_cluster,gadget_path=fl_param['gadget_path'],qsub_output_path=fl_param['qsub_output_path'])

######################## plot shell arrangement ########################
fl.plot_shell_arrange.plot_shell_arrange(fl_param, root_local)

