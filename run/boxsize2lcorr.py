import numpy as np
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
from matplotlib import patches
import os
import QingLongTool as ql
import camb

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

######################## Plot grid ########################
os.makedirs(os.path.join(root_local, 'nbody_sim'), exist_ok=True)
os.makedirs(os.path.join(root_local, 'lensing'), exist_ok=True)
os.makedirs(os.path.join(root_local, 'nbody_sim/pdf'), exist_ok=True)

pars = camb.CAMBparams()
pars.set_cosmology(H0=ql_param['H0'], ombh2=ql_param['ombh2'], omch2=ql_param['omch2'], 
    tau=ql_param['tau'], mnu=ql_param['mnu'], omk=ql_param['omk'])
pars.InitPower.set_params(As=ql_param['As'], ns=ql_param['ns'])
results = camb.get_background(pars)
chistar = (results.conformal_time(0) - results.tau_maxvis)*ql_param['H0']/100

boxsize_list = np.loadtxt(f"{os.path.join(root_local, f'nbody_sim/boxsize_list.txt')}")
num_box = len(boxsize_list)

print(boxsize_list)
l_list = []
z_list = []
for idx in range(len(boxsize_list)):
    intervals = np.load(f"{os.path.join(root_local, f'shells/shell_intervals_b{idx+1}.npy')}")
    redshifts = (1/(np.loadtxt(f"{os.path.join(root_local, f'nbody_sim/outputs_b{idx+1}.txt')}"))-1)
    intervals_mid = np.mean(intervals,axis = 1)

    for j in reversed(range(len(intervals_mid))):
        circum = 2*np.pi*intervals_mid[j]
        angle = boxsize_list[idx]/circum*360
        l = 180/angle
        l_list.append(l)
        z_list.append(redshifts[j])

        print(redshifts[j],intervals_mid[j])


print((results.conformal_time(0) - results.tau_maxvis))

plt.plot(z_list,l_list)
plt.ylabel(r'$\ell^{eff}$')
plt.xlabel(r'$redshift$')
plt.title(r'boxsize corresponds to $\ell$ at snapped redshift')
plt.savefig(os.path.join(root_local, f'lensing/boxsize2lcorr.png'))
plt.show()

