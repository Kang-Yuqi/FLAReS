import FlaresTool as fl
import camb
from mpi4py import MPI

######################## load param #########################
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-p', '--param-file', default='../param/param.ini', type=str, help='Path to the parameter file')
parser.add_argument('-s', '--random-seed', default='0', type=int, help='random seed for shells rotation')

################################################ run ################################################
args = parser.parse_args()
random_seed = args.random_seed
fl_param = fl.read_config(args.param_file)
mpi = True
######################## cosmology with CAMB ##################
pars = camb.CAMBparams()
pars.set_cosmology(H0=fl_param['H0'], ombh2=fl_param['ombh2'], omch2=fl_param['omch2'], 
    tau=fl_param['tau'], mnu=fl_param['mnu'], omk=fl_param['omk'])
pars.InitPower.set_params(As=fl_param['As'], ns=fl_param['ns'])
results = camb.get_results(pars)

cache_dir = fl.lensing_post_Born.lens_shell_prepare(fl_param, pars=pars, random_seed=random_seed, cache_dir=None, mpi=mpi,new=False)
fl.lensing_post_Born.kappa_Born(fl_param, pars=pars, cache_dir=cache_dir, random_seed=random_seed, mpi=mpi, new=False)
fl.lensing_post_Born.kappa_ll(fl_param, pars=pars, cache_dir=cache_dir, random_seed=random_seed, mpi=mpi, new=False)
fl.lensing_post_Born.kappa_geo(fl_param, pars=pars, cache_dir=cache_dir, random_seed=random_seed, mpi=mpi, new=False)

if mpi:
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    comm.Barrier()
    if rank == 0:
        import shutil
        shutil.rmtree(cache_dir)
else:
    shutil.rmtree(cache_dir)

