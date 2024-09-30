import numpy as np
import healpy as hp
import collections
from tqdm import tqdm
# from memory_profiler import profile

def readHeader(snapshot):
    import readgadget

    header   = readgadget.header(snapshot)
    BoxSize  = header.boxsize      # Mpc/h
    Nall     = header.nall         # Total number of particles
    Masses   = header.massarr*1e10 # Masses of the particles in Msun/h
    Omega_m  = header.omega_m      # value of Omega_m
    Omega_l  = header.omega_l      # value of Omega_l
    h        = header.hubble       # value of h
    redshift = header.redshift     # redshift of the snapshot
    Hubble   = 100.0*np.sqrt(Omega_m*(1.0+redshift)**3+Omega_l) # Value of H(z) in km/s/(Mpc/h)
    
    return BoxSize, Nall, Masses, Omega_m, Omega_l, h, redshift, Hubble


def calculate_vertex_distances(center, boxsize):
    vertices = np.array([[1, 0, 1], [1, 0, -1], [-1, 0, 1], [-1, 0, -1],
                        [0, 1, 1], [0, 1, -1], [0, -1, 1], [0, -1, -1],
                        [1, 1, 0], [1, -1, 0], [-1, 1, 0], [-1, -1, 0],
                        [1, 0, 0], [-1, 0, 0], 
                        [0, 1, 0], [0, -1, 0],
                        [0, 0, 1], [0, 0, -1],
                        [1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1],
                        [-1, 1, 1], [-1, 1, -1], [-1, -1, 1], [-1, -1, -1]])
    
    vertices = center + 0.5 * boxsize * vertices
    distances = np.linalg.norm(vertices, axis=1)
    
    # If the center is within the box, set the minimum distance to 0
    if np.all(np.abs(center) <= 0.5 * boxsize):
        min_distance = 0
    else:
        min_distance = np.min(distances)
    
    max_distance = np.max(distances)
    
    return min_distance, max_distance
    
def calculate_box_indices(steps, boxsize, shell_min, shell_max):                
    steps = np.asarray(steps)
    axis_shift_list = np.stack(np.meshgrid(steps, steps, steps), -1).reshape(-1, 3)

    inner_dis = []
    outer_dis = []
    for center in axis_shift_list:
        min_dist, max_dist = calculate_vertex_distances(center*boxsize, boxsize)
        inner_dis.append(min_dist)
        outer_dis.append(max_dist)

    inner_dis = np.array(inner_dis)
    outer_dis = np.array(outer_dis)

    return axis_shift_list[np.where((inner_dis < shell_max) & (outer_dis > shell_min))]

def box_shift_pos(pos, boxsize=None, axis_shift=None):
    axis_shift_center = axis_shift.astype(np.float64) - 0.5 # the center of Nbody box is [0.5,0.5,0.5]*boxsize
    shifted_pos = np.copy(pos)
    shifted_pos[:, 0] += axis_shift_center[0] * boxsize
    shifted_pos[:, 1] += axis_shift_center[1] * boxsize
    shifted_pos[:, 2] += axis_shift_center[2] * boxsize
    return shifted_pos

def scatter_axis_shift_list(comm, axis_shift_list):
    num_box_require = len(axis_shift_list)
    comm_size = comm.Get_size()
    mpi_split_step = int(num_box_require / comm_size) + 1
    mpi_split_axis_shift_list = [axis_shift_list[i:i+mpi_split_step] for i in range(0, num_box_require, mpi_split_step)]

    if len(mpi_split_axis_shift_list) < comm_size:
        mpi_split_axis_shift_list.extend([[]] * (comm_size - len(mpi_split_axis_shift_list)))

    return mpi_split_axis_shift_list

def gather_results_chunk(comm, proj_map_local, num_particle_local, chunk_size=300):
    num_chunks = len(proj_map_local) // chunk_size + (1 if len(proj_map_local) % chunk_size else 0)
    
    if comm.Get_rank() == 0:
        proj_map = np.zeros_like(proj_map_local)
    combine_num_particle = comm.gather(num_particle_local, root=0) 
    
    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = start_idx + chunk_size
        
        chunk_proj_map = proj_map_local[start_idx:end_idx]
        combine_proj_map_chunk = comm.gather(chunk_proj_map, root=0)

        if comm.Get_rank() == 0:
            proj_map[start_idx:end_idx] += sum(combine_proj_map_chunk)
  
    if comm.Get_rank() == 0:
        num_particle = sum(combine_num_particle)
        return proj_map, num_particle
    else:
        return None, None
        
# @profile
def singlebox2shell_project_mpi(comm, box_base_name, shell_maxmin,Nside_map=None,ptype=[1], batch_size = 1e7):
    batch_size = int(batch_size)
    import readgadget

    boxsize, num_par_box, Masses, Omega_m, Omega_l, h, redshift, Hubble = readHeader(box_base_name)
    shell_min, shell_max = shell_maxmin
    boxsize = boxsize.astype(np.float64)

    comm_rank = comm.Get_rank()

    if comm_rank == 0:
        layer_num = int(shell_max // boxsize + 1)
        steps = np.arange(-layer_num, layer_num + 1)
        axis_shift_list = calculate_box_indices(steps, boxsize, shell_min, shell_max)
        num_box_shell = len(axis_shift_list)
        print(f'{len(axis_shift_list)} boxes stacked together! shell thickness: {(shell_max-shell_min)} Mpc/h')
        mpi_split_axis_shift_list = scatter_axis_shift_list(comm, axis_shift_list)
    else:
        mpi_split_axis_shift_list = 0

    axis_shift_list = comm.scatter(mpi_split_axis_shift_list, root=0) # scattering shift_list to each cores

    num_particle_local = 0
    npix = hp.nside2npix(Nside_map)
    proj_map_local = np.zeros(npix)
    pos_base = readgadget.read_block(box_base_name, "POS ", ptype).astype(np.float64)

    rank_str = f'{comm_rank:03}'
    with tqdm(total=len(axis_shift_list), desc=f'Rank {rank_str} progress', position=comm_rank, leave=True) as pbar:

        for axis_shift in axis_shift_list:
            proj_map_local_eachbox = np.zeros(npix)
            shell_par_num = 0
            for i in range(0, len(pos_base), batch_size):
                pos_base_batch = pos_base[i:i + batch_size]

                normalized_pos_base_batch = pos_base_batch / boxsize
                normalized_boxsize = boxsize / boxsize
                normalized_shell_min = shell_min / boxsize
                normalized_shell_max = shell_max / boxsize

                shell_pos = box_shift_pos(normalized_pos_base_batch, boxsize=normalized_boxsize, axis_shift=axis_shift)

                R = np.sqrt(np.sum(shell_pos ** 2, axis=1))

                index = np.where((R < normalized_shell_min)|(R >= normalized_shell_max))
                shell_pos = np.delete(shell_pos, index, axis=0)
                R = np.delete(R, index, axis=0)

                particles_theta = np.arccos(np.clip(shell_pos[:, 2] / R, -1, 1))
                particles_phi = np.arctan2(shell_pos[:, 1], shell_pos[:, 0])            

                pix = hp.pixelfunc.ang2pix(Nside_map, particles_theta, particles_phi)
                proj_map_local_eachbox[list(collections.Counter(pix).keys())] += list(collections.Counter(pix).values())
                shell_par_num += len(shell_pos)

            proj_map_local += proj_map_local_eachbox
            num_particle_local += shell_par_num

            pbar.update(1)

    comm.Barrier()
    proj_map, num_particle = gather_results_chunk(comm, proj_map_local, num_particle_local)

    if comm.Get_rank() == 0:
        shell_volume = 4/3*np.pi*(shell_max**3-shell_min**3)
        shell_volume_pix = shell_volume/(Nside_map**2 * 12)
        map_contrast = proj_map/(shell_volume_pix*(num_particle/shell_volume)) -1
        return map_contrast,{"num_particles_pix": shell_volume_pix*(num_particle/shell_volume), "num_box_shell": num_box_shell}
    else:
        return None,None
