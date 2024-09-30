import configparser
import ast
import os 
from . import initial_power_spectrum
from . import gen_Gadget
from . import plot_shell_arrange
from . import box2dens
from . import analytical_power_spectrum
from . import lensing
from . import nbody_analysis
from . import nG_analysis
from . import needlets
from . import shell_thickness_corr
from . import lensing_post_Born

def read_config(filepath):

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"{filepath} not exists")

    config = configparser.ConfigParser()
    config.optionxform = str
    config.read(filepath)

    params = {}

    sections = {
        'path': str,
        'cosmology_model': float,
        'shell_arrange': None,
        'Nbody': None,
        'shell_projection': None,
        'lensing': None,
        'analysis': None
    }

    for section, dtype in sections.items():
        if config.has_section(section):
            for option in config.options(section):
                value = config[section][option]
                if dtype == float:
                    params[option] = float(value)
                elif dtype == str:
                    params[option] = value
                else:
                    try:
                        params[option] = int(value)
                    except ValueError:
                        try:
                            params[option] = ast.literal_eval(value)
                        except (ValueError, SyntaxError):
                            params[option] = value
        else:
            print(f"Warning: Section '{section}' not found.")

    return params

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