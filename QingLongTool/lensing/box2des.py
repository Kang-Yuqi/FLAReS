import numpy as np
import healpy as hp
import collections

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


def box_shift_pos(x, boxsize=None, center=None, shift_unit=0):
    if center is None:
        center = [0, 0, 0]

    if all(val == 0 for val in center):
        x[:, 0] -= boxsize / 2
        x[:, 1] -= boxsize / 2
        x[:, 2] -= boxsize / 2
    elif shift_unit == 0:
        x[:, 0] += center[0] - boxsize / 2
        x[:, 1] += center[1] - boxsize / 2
        x[:, 2] += center[2] - boxsize / 2
    elif shift_unit == 1:
        x[:, 0] -= boxsize / 2 - boxsize * center[0]
        x[:, 1] -= boxsize / 2 - boxsize * center[1]
        x[:, 2] -= boxsize / 2 - boxsize * center[2]
    else:
        print('Shift unit not allowed!')

    return x


def rotmatrix(rtheta, axis=None):
    if axis == 'z':
        matrix = np.array([
            [np.cos(rtheta), -np.sin(rtheta), 0],
            [np.sin(rtheta), np.cos(rtheta), 0],
            [0, 0, 1]
        ])
    elif axis == 'y':
        matrix = np.array([
            [np.cos(rtheta), 0, np.sin(rtheta)],
            [0, 1, 0],
            [-np.sin(rtheta), 0, np.cos(rtheta)]
        ])
    elif axis == 'x':
        matrix = np.array([
            [1, 0, 0],
            [0, np.cos(rtheta), -np.sin(rtheta)],
            [0, np.sin(rtheta), np.cos(rtheta)]
        ])
    else:
        raise ValueError(f"Invalid axis: {axis}. Choose 'x', 'y', or 'z'.")

    return matrix


def inner_outer(x):
    if x >= 0:
        x1, x2 = x + 0.5, x - 0.5
    else:
        x1, x2 = x - 0.5, x + 0.5

    return x1, x2


def xyz2spherical(xyz):
    spher = np.zeros(xyz.shape)
    x2y2 = xyz[:, 0]**2 + xyz[:, 1]**2
    spher[:, 0] = np.sqrt(x2y2 + xyz[:, 2]**2)
    spher[:, 1] = np.arctan2(np.sqrt(x2y2), xyz[:, 2])  # z axis down
    spher[:, 2] = np.arctan2(xyz[:, 1], xyz[:, 0])  # x axis clockwise
    return spher


def spherical2xyz(spher):
    xyz = np.zeros(spher.shape)
    xyz[:, 0] = spher[:, 0] * np.sin(spher[:, 1]) * np.cos(spher[:, 2])
    xyz[:, 1] = spher[:, 0] * np.sin(spher[:, 1]) * np.sin(spher[:, 2])
    xyz[:, 2] = spher[:, 0] * np.cos(spher[:, 1])
    return xyz

def calculate_box_indices(steps, boxsize,shell_min, shell_max):
    axis_shift_list = []
    axis_shift_outer_list = []
    axis_shift_inner_list = []

    for x in steps:
        x1, x2 = inner_outer(x)
        for y in steps:
            y1, y2 = inner_outer(y)
            for z in steps:
                z1, z2 = inner_outer(z)
                axis_shift_list.append([x, y, z])
                axis_shift_outer_list.append([x1, y1, z1])
                axis_shift_inner_list.append([x2, y2, z2])

    axis_shift_list = np.asarray(axis_shift_list) + 0.5
    outer_coor = (np.asarray(axis_shift_outer_list) + 0.5) * boxsize
    inner_coor = (np.asarray(axis_shift_inner_list) + 0.5) * boxsize

    outer_dis = np.sqrt(np.sum(outer_coor**2, axis=1))
    inner_dis = np.sqrt(np.sum(inner_coor**2, axis=1))


    return axis_shift_list[np.where((inner_dis < shell_max) & (outer_dis > shell_min))]


def scatter_axis_shift_list(comm, axis_shift_list):
    num_box_require = len(axis_shift_list)
    comm_size = comm.Get_size()
    mpi_split_step = int(num_box_require / comm_size) + 1
    mpi_split_axis_shift_list = [axis_shift_list[i:i+mpi_split_step] for i in range(0, num_box_require, mpi_split_step)]

    if len(mpi_split_axis_shift_list) < comm_size:
        mpi_split_axis_shift_list.extend([[]] * (comm_size - len(mpi_split_axis_shift_list)))

    return mpi_split_axis_shift_list


def process_particles_in_box(pos_base, boxsize, axis_shift, rotate_shell,shell_min, shell_max):
    pos_shifted = box_shift_pos(pos_base, boxsize=boxsize, center=axis_shift, shift_unit=1)
    spher_pos_shifted = xyz2spherical(pos_shifted)
    index = np.where((spher_pos_shifted[:, 0] < shell_min) | (spher_pos_shifted[:, 0] > shell_max))
    shell_spher_pos = np.delete(spher_pos_shifted, index, axis=0)
    shell_pos = spherical2xyz(shell_spher_pos)

    for i, angle in enumerate(rotate_shell):
        if angle:
            axes = ['x', 'y', 'z']
            matrix = rotmatrix(angle, axis=axes[i])
            shell_pos = np.einsum('ij,...j', matrix, shell_pos)

    return xyz2spherical(shell_pos)

def gather_results(comm, proj_map_local, num_particle_local):
    combine_proj_map = comm.gather(proj_map_local, root=0)
    combine_num_particle = comm.gather(num_particle_local, root=0)
    
    if comm.Get_rank() == 0:
        proj_map = sum(combine_proj_map)
        num_particle = sum(combine_num_particle)
        return proj_map, num_particle
    else:
        return None, None

def singlebox2shell_project_mpi(comm, box_base_name, shell_maxmin, 
    rotate_shell=[0, 0, 0], ptype=[1], Nside_map=None):
    import readgadget

    boxsize, num_par_box, Masses, Omega_m, Omega_l, h, redshift, Hubble = readHeader(box_base_name)
    shell_min, shell_max = shell_maxmin
    layer_num = shell_max // boxsize + 1

    comm_rank = comm.Get_rank()

    if comm_rank == 0:
        steps = np.arange(layer_num * 2) - layer_num
        axis_shift_list = calculate_box_indices(steps, boxsize,shell_min, shell_max)
        print(f'{len(axis_shift_list)} boxes stacked together! shell thickness: {(shell_max-shell_min)/h} Mpc')

        mpi_split_axis_shift_list = scatter_axis_shift_list(comm, axis_shift_list)
    else:
            mpi_split_axis_shift_list = 0

    axis_shift_list = comm.scatter(mpi_split_axis_shift_list, root=0) # scattering shift_list to each cores


    num_particle_local = 0
    proj_map_local = np.zeros(Nside_map**2 * 12)

    for axis_shift in axis_shift_list:
        pos_base = readgadget.read_block(box_base_name, "POS ", ptype)
        shell_spher_pos_local = process_particles_in_box(pos_base, boxsize, axis_shift, rotate_shell,shell_min, shell_max)
        shell_sphr_2dpos_local = shell_spher_pos_local[:, 1:]
        pix = hp.pixelfunc.ang2pix(Nside_map, shell_sphr_2dpos_local[:, 0], shell_sphr_2dpos_local[:, 1])
        proj_map_local[list(collections.Counter(pix).keys())] += list(collections.Counter(pix).values())
        num_particle_local += len(shell_spher_pos_local)

    combine_proj_map = comm.gather(proj_map_local, root=0)
    combine_num_particle = comm.gather(num_particle_local, root=0)

    proj_map, num_particle = gather_results(comm, proj_map_local, num_particle_local)


    if comm.Get_rank() == 0:
        shell_volume = 4/3*np.pi*(shell_max**3-shell_min**3)
        shell_volume_pix = shell_volume/(Nside_map**2 * 12)

        map_contrast = proj_map/(shell_volume_pix*(num_particle/shell_volume)) -1
        return map_contrast