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
os.makedirs(os.path.join(root_local, 'nbody_sim/pdf'), exist_ok=True)

pars = camb.CAMBparams()
pars.set_cosmology(H0=ql_param['H0'], ombh2=ql_param['ombh2'], omch2=ql_param['omch2'], 
    tau=ql_param['tau'], mnu=ql_param['mnu'], omk=ql_param['omk'])
pars.InitPower.set_params(As=ql_param['As'], ns=ql_param['ns'])
results = camb.get_background(pars)
chistar = (results.conformal_time(0) - results.tau_maxvis)*ql_param['H0']/100

boxsize_list = np.loadtxt(f"{os.path.join(root_local, f'nbody_sim/boxsize_list.txt')}")
num_box = len(boxsize_list)

intervals_list = []
for idx in range(num_box):
    intervals = np.load(f"{os.path.join(root_local, f'shells/shell_intervals_b{idx+1}.npy')}")[::-1]
    intervals_list.append(intervals)

N = ql_param['Gaussian_shell_steps']
points = np.linspace(intervals[-1][1],chistar, N+1)
intervals_G = [[points[i], points[i+1]] for i in range(N)]
intervals_list.append(intervals_G)
boxsize_list = np.append(boxsize_list,1e6)

num_box = num_box+1

max_radi = intervals_list[num_box-1][-1][1]
fig = plt.figure(figsize=(6, 6))
axes = [ql.nbody.plot_arrange.create_ax(fig,max_radi) for _ in range(num_box)]

for idx in range(num_box):
    if idx == 0:
        draw_func = ql.nbody.plot_arrange.draw_rotated_grid_circle
        kwargs = {
            "rotation_angle": 45,
            "colors": ['yellow', 'blue']
        }
    else:
        draw_func = ql.nbody.plot_arrange.draw_rotated_grid_annulus
        kwargs = {
            "width": intervals_list[idx][-1][1] - intervals_list[idx][0][0],
            "rotation_angle": 0 if idx % 2 == 1 else 45,
            "colors": ['red', 'green'] if idx % 2 == 1 else ['yellow', 'blue']
        }

    draw_func(axes[idx],
              grid_size=boxsize_list[idx],
              radius=intervals_list[idx][-1][1],
              intervals=intervals_list[idx],
              **kwargs)

tick_values = [0] + [interval[-1][1] for interval in intervals_list]
tick_values_label = np.round(results.redshift_at_comoving_radial_distance(np.array(tick_values) / (ql_param['H0'] / 100)), 2)
axes[num_box-1].xaxis.set_visible(True)
axes[num_box-1].set_xticks(tick_values)
axes[num_box-1].set_xticklabels(tick_values_label, rotation=30)
axes[num_box-1].set_xlabel('redshift')
axes[num_box-1].xaxis.set_label_coords(0.5, -0.1)

plt.savefig(os.path.join(root_local, f'nbody_sim/nbody_arrange.png'))
plt.savefig(os.path.join(root_local, f'nbody_sim/pdf/nbody_arrange.pdf'))
plt.close()