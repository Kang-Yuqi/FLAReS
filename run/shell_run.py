import FlaresTool as fl
import os
import numpy as np
import mpi4py.MPI as MPI
import datetime
import healpy as hp
import readgadget

######################## load param #########################
import argparse
parser = argparse.ArgumentParser(description='Process N-body simulation shells.')
parser.add_argument('-p', '--param-file', default='../param/param.ini', type=str, help='Path to the parameter file')
parser.add_argument('-s', '--start_idx', default= -1, type=int, help='The index of simulation that the projection process start from, default from the largest box.')

parser.add_argument('-i', '--idx', default=None, type=int, help='Specific shell index')
parser.add_argument('-c', '--counter', default=None, type=int, help='Specific counter')

args = parser.parse_args()
start_idx = args.start_idx
idx_choose = args.idx
counter_choose = args.counter
fl_param = fl.read_config(args.param_file)
n = sum(fl_param['num_shells'])
if start_idx == -1:
    start_idx = fl_param['num_box']
    
######################## path #########################
root_local = fl_param['root_local_base'] + fl_param['root_unify']
root_cluster = fl_param['root_cluster_base'] + fl_param['root_unify']
if fl_param['current_base'] == 'cluster':
    root_local = root_cluster
elif fl_param['current_base'] == 'local':
    pass 
else:
    raise TypeError("current_base should be 'local' or 'cluster'.")
snapshot_root = root_local + fl_param['Nbody_folder']

####################### Cutting shell from by stacking Nbody box ########################
#######################    and projecting particles to full sky map ########################
#set up mpi
comm = MPI.COMM_WORLD
comm_rank = comm.Get_rank()
comm_size = comm.Get_size()

if idx_choose:
    if comm_rank == 0:
        with open(os.path.join(root_local, f'shells/projection_progress_update.txt'), 'a') as file:
            pass
    idx_list = [idx_choose]
else:
    # make an empty progress recorder
    if comm_rank == 0:
        with open(os.path.join(root_local, f'shells/projection_progress_update.txt'), 'w') as file:
            pass
    idx_list = reversed(range(1, start_idx + 1))

# projection process start, from the largest box to the smallest
for idx in idx_list:
    if comm_rank == 0:
        os.makedirs(os.path.join(root_local, f'shells/nbody{idx}'), exist_ok=True)

    intervals = np.load(os.path.join(root_local, f'shells/shell_intervals_{n}Boxshells_b{idx}.npy'))

    if counter_choose:
        counter_list = [counter_choose]
    else:
        counter_list = range(len(intervals))

    if comm_rank == 0:
        redshifts = []

    for counter in counter_list:
        box_base_name = os.path.join(snapshot_root, f'nbody{idx}/snapshot_{counter:03}.hdf5')
        if comm_rank == 0:
            header = readgadget.header(box_base_name)
            redshift = header.redshift
            redshifts.append(redshift)

        if comm_rank == 0: 
            current_time = datetime.datetime.now()
            formatted_time = current_time.strftime('%Y-%m-%d %H:%M:%S')
            with open(os.path.join(root_local, f'shells/projection_progress_update.txt'), 'a') as file:
                file.write(f"[{formatted_time}] Processing {snapshot_root}/nbody{idx}/snapshot_{counter:03}.hdf5 \n")

        map_contrast, info = fl.box2dens.singlebox2shell_project_mpi(comm, box_base_name, intervals[counter], Nside_map=fl_param['Nside_shell'])

        if comm_rank == 0:
            hp.fitsfunc.write_map(os.path.join(root_local, f"shells/nbody{idx}/contrast_map_{fl_param['Nbody_folder']}_Nside{fl_param['Nside_shell']}_{counter:03}.fits"), map_contrast, overwrite=True)
            
            num_particles_pix = info["num_particles_pix"]
            num_box_shell = info["num_box_shell"]

            current_time = datetime.datetime.now()
            formatted_time = current_time.strftime('%Y-%m-%d %H:%M:%S')
            with open(os.path.join(root_local, f'shells/projection_progress_update.txt'), 'a') as file:
                file.write(f"{num_box_shell} boxes stacked together, with an average of {num_particles_pix:.2f} particles per pixel.\n")
                file.write(f"[{formatted_time}] Contrast map saved as {root_local}shells/nbody{idx}/contrast_map_{fl_param['Nbody_folder']}_Nside{fl_param['Nside_shell']}_{counter:03}.fits \n")

    if comm_rank == 0:
        if idx_choose is None:
            np.savetxt(os.path.join(root_local, f'shells/shell_Nbody_redshifts_{n}Boxshells_b{idx}.txt'), redshifts)