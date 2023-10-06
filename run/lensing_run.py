import QingLongTool as ql
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import os 
import numpy as np
import healpy as hp
from astropy.constants import c
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

######################## lensing ########################
os.makedirs(os.path.join(root_local, 'lensing'), exist_ok=True)
os.makedirs(os.path.join(root_local, 'lensing/pdf'), exist_ok=True)

pars = camb.CAMBparams()
pars.set_cosmology(H0=ql_param['H0'], ombh2=ql_param['ombh2'], omch2=ql_param['omch2'], 
	tau=ql_param['tau'], mnu=ql_param['mnu'], omk=ql_param['omk'])
pars.InitPower.set_params(As=ql_param['As'], ns=ql_param['ns'])
results = camb.get_background(pars)
chistar = (results.conformal_time(0) - results.tau_maxvis)*ql_param['H0']/100

map_sum = np.zeros(ql_param['Nside_map']**2*12)
boxsize_list = np.loadtxt(f"{os.path.join(root_local, f'nbody_sim/boxsize_list.txt')}")
lmax = 3*ql_param['Nside_map']-1
h = ql_param['H0']/100

cl_map_sum_list = []
redshift_end_list = []
redshift_mid_list = []
nG_list = []
#### lensing from Nbody simulation ####
for idx in range(len(boxsize_list)):
	intervals = np.load(f"{os.path.join(root_local, f'shells/shell_intervals_b{idx+1}.npy')}")
	redshifts = (1/(np.loadtxt(f"{os.path.join(root_local, f'nbody_sim/outputs_b{idx+1}.txt')}"))-1)

	cl_matter_list = ql.lensing.analytical.Cl_matter(redshifts,lmin=ql_param['lmin'],lmax=lmax,
	    kmax=1e2,nonlinear=True,H0=ql_param['H0'], ombh2=ql_param['ombh2'], omch2=ql_param['omch2'], 
	    tau=ql_param['tau'], mnu=ql_param['mnu'], omk=ql_param['omk'],As=ql_param['As'], ns=ql_param['ns'])

	for counter in reversed(range(len(intervals))):
		redshift = redshifts[counter]
		box_base_name = os.path.join(ql_param['snapshot_root'], f'nbody{idx+1}/snapshot_{counter:03}.hdf5')
		redshift_end = results.redshift_at_comoving_radial_distance(intervals[counter][1]/(ql_param['H0']/100))
		[ls_matter,ks_matter,cl_matter] = cl_matter_list[counter]

		alm_G = hp.sphtfunc.synalm(cl_matter, lmax=lmax)/(np.mean(intervals[counter])**2*(intervals[counter][1]-intervals[counter][0]))**0.5
		map_contrast = hp.fitsfunc.read_map(os.path.join(root_local, f'shells/nbody{idx+1}/contrast_map_{counter:03}.fits'))
		alm_contrast = hp.sphtfunc.map2alm(map_contrast, lmax=lmax)

		# replace large scale with Gaussian realizations
		fl_G = np.ones(lmax+1)
		fl_G[ql_param['l_corr']:]=0
		alm_G_filter = hp.sphtfunc.almxfl(alm_G, fl_G, inplace=False)

		fl_contrast = np.ones(lmax+1)
		fl_contrast[:ql_param['l_corr']]=0
		alm_contrast_filter = hp.sphtfunc.almxfl(alm_contrast, fl_contrast, inplace=False)

		alm_corr = alm_contrast_filter + alm_G_filter
		map_contrast = hp.sphtfunc.alm2map(alm_corr, ql_param['Nside_map'], lmax=lmax)

		# add up maps (Born approx)
		chi = np.mean(intervals[counter])
		win = (chistar-chi)*chi/chistar
		map_sum += map_contrast*win/(1/(1+redshift))*(intervals[counter][1]-intervals[counter][0])

		# NG
		alm_sum = hp.sphtfunc.map2alm(map_sum*(3*ql_param['H0']**2*(ql_param['ombh2']+ql_param['omch2'])/h**2/(2*c.to('km/s').value**2)/h**2), lmax=lmax)
		[sigma,sigma1,S,S1,S2,K,K1,K12,K22] = ql.NG.SK_param.S_K_parameter(alm_sum,ql_param['Nside_map'],lmax)

		nG_list.append([sigma,sigma1,S,S1,S2,K,K1,K12,K22])
		cl_map_sum_list.append(hp.sphtfunc.anafast(map_sum, lmax=3*ql_param['Nside_map']-1))
		redshift_end_list.append(redshift_end)
		redshift_mid_list.append(redshift)

# #### lensing from Gaussian simulation ####
intervals = np.load(f"{os.path.join(root_local, f'shells/shell_intervals_b{len(boxsize_list)}.npy')}")
N = ql_param['Gaussian_shell_steps']
points = np.linspace(intervals[0][1],chistar, N+1)
intervals_G = [[points[i], points[i+1]] for i in range(N)]
midpoints_G = [np.mean([points[i],points[i+1]]) for i in range(N)]

midpoints_redshift_G = results.redshift_at_comoving_radial_distance(np.asarray(midpoints_G)/(ql_param['H0']/100))

cl_matter_list = ql.lensing.analytical.Cl_matter(midpoints_redshift_G,lmin=ql_param['lmin'],lmax=lmax,
    kmax=1e2,nonlinear=True,H0=ql_param['H0'], ombh2=ql_param['ombh2'], omch2=ql_param['omch2'], 
    tau=ql_param['tau'], mnu=ql_param['mnu'], omk=ql_param['omk'],As=ql_param['As'], ns=ql_param['ns'])

for counter in range(len(midpoints_redshift_G)):
	redshift = midpoints_redshift_G[counter]
	redshift_end = results.redshift_at_comoving_radial_distance(intervals_G[counter][1]/(ql_param['H0']/100))
	[ls_matter,ks_matter,cl_matter] = cl_matter_list[counter]

	map_G = hp.sphtfunc.synfast(cl_matter, ql_param['Nside_map'], lmax=lmax)/(np.mean(intervals_G[counter])**2*(intervals_G[counter][1]-intervals_G[counter][0]))**0.5
	chi = np.mean(intervals_G[counter])
	win = (chistar-chi)*chi/chistar
	map_sum += map_G*win/(1/(1+redshift))*(intervals_G[counter][1]-intervals_G[counter][0])

	# NG
	alm_sum = hp.sphtfunc.map2alm(map_sum*(3*ql_param['H0']**2*(ql_param['ombh2']+ql_param['omch2'])/h**2/(2*c.to('km/s').value**2)/h**2), lmax=lmax)
	[sigma,sigma1,S,S1,S2,K,K1,K12,K22] = ql.NG.SK_param.S_K_parameter(alm_sum,ql_param['Nside_map'],lmax)
	
	nG_list.append([sigma,sigma1,S,S1,S2,K,K1,K12,K22])
	cl_map_sum_list.append(hp.sphtfunc.anafast(map_sum, lmax=3*ql_param['Nside_map']-1))
	redshift_end_list.append(redshift_end)
	redshift_mid_list.append(redshift)

######################## result ########################
map_sum *= 3*ql_param['H0']**2*(ql_param['ombh2']+ql_param['omch2'])/h**2/(2*c.to('km/s').value**2)/h**2 # last "/h^2", make Mpc/h to Mpc for lensing
hp.fitsfunc.write_map(os.path.join(root_local, f'lensing/convergence_map.fits'), map_sum,overwrite=True)

cl_kappa_sim = hp.sphtfunc.anafast(map_sum, lmax=lmax)
cl_kappa_list = np.asarray(cl_map_sum_list)*(3*ql_param['H0']**2*(ql_param['ombh2']+ql_param['omch2'])/h**2/(2*c.to('km/s').value**2)/h**2)**2
nG_array = np.array(nG_list)

np.save(f"{os.path.join(root_local, f'lensing/cl_kappa.npy')}",cl_kappa_sim)
np.save(f"{os.path.join(root_local, f'lensing/redshift_end_list.npy')}",redshift_end_list)
np.save(f"{os.path.join(root_local, f'lensing/redshift_mid_list.npy')}",redshift_mid_list)
np.save(f"{os.path.join(root_local, f'lensing/nG_evolution.npy')}",nG_array)
np.save(f"{os.path.join(root_local, f'lensing/cl_kappa_list.npy')}",cl_kappa_list)

#### analytical prediction from CAMB ####
[ls_kappa,cl_kappa_camb] = ql.lensing.analytical.lensing_kappa_CAMB(lmax=lmax,H0=ql_param['H0'], ombh2=ql_param['ombh2'], 
    omch2=ql_param['omch2'], tau=ql_param['tau'], mnu=ql_param['mnu'], omk=ql_param['omk'],As=ql_param['As'], ns=ql_param['ns'])

####################### plot ########################

#### lensing map ####
map_sum = hp.fitsfunc.read_map(os.path.join(root_local, f'lensing/convergence_map.fits'))

hp.visufunc.mollview(map=map_sum)
plt.savefig(os.path.join(root_local, f'lensing/convergence_map.png'))
plt.savefig(os.path.join(root_local, f'lensing/pdf/convergence_map.pdf'))
plt.close()

#### lensing power spectrum ####
cl_kappa_sim = np.load(f"{os.path.join(root_local, f'lensing/cl_kappa.npy')}")

plt.loglog(ls_kappa, cl_kappa_sim)
plt.loglog(ls_kappa, cl_kappa_camb)
plt.xlim([ql_param['lmin'], max(ls_kappa)])
plt.legend(['Nbody', 'CAMB hybrid'])
plt.ylabel(r'$[\ell(\ell+1)]^2C_\ell^{\phi\phi}/4$')
plt.xlabel(r'$\ell$')
plt.axvline(x=ql_param['l_corr'], color='gray',linestyle='--')
plt.annotate(r'$\ell_{corr}$=%s' %ql_param['l_corr'], xy=(ql_param['l_corr'], max(cl_kappa_camb)*0.2), color='black', 
             xytext=(ql_param['l_corr']+20, max(cl_kappa_camb)*0.3), arrowprops=dict(facecolor='black', arrowstyle="->"),
             horizontalalignment='center')
plt.savefig(os.path.join(root_local, f'lensing/convergence_cl.png'))
plt.savefig(os.path.join(root_local, f'lensing/pdf/convergence_cl.pdf'))
plt.close()

#### lensing power spectrum evolution####
import matplotlib.cm as cm
from matplotlib.colors import LogNorm
from matplotlib.ticker import FormatStrFormatter, NullLocator

cl_kappa_list = np.load(f"{os.path.join(root_local, f'lensing/cl_kappa_list.npy')}")
redshift_list = np.load(f"{os.path.join(root_local, f'lensing/redshift_mid_list.npy')}")

cmap = cm.get_cmap('rainbow', len(redshift_list))
norm = LogNorm(vmin=min(redshift_list), vmax=max(redshift_list))

step = 5
fig, ax = plt.subplots()
for i, redshift in enumerate(redshift_list[::step]):
	ax.loglog(ls_kappa[ql_param['lmin']:-1], cl_kappa_list[step*i][ql_param['lmin']:-1], color=cmap(norm(redshift)))

sm = cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, ticks=redshift_list[::step])
cbar.set_label(r'$z_{<}$', rotation=270, labelpad=15)
cbar.ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
cbar.ax.yaxis.set_minor_locator(NullLocator())
plt.ylabel(r'$[\ell(\ell+1)]^2C_\ell^{\phi\phi}/4$')
plt.xlabel(r'$\ell$')
plt.title('Lensing convergence power spectrum')
plt.savefig(os.path.join(root_local, f'lensing/convergence_cl_evolution.png'))
plt.savefig(os.path.join(root_local, f'lensing/pdf/convergence_cl_evolution.pdf'))
plt.close()

#### lensing NG evolution####
nG_array = np.load(f"{os.path.join(root_local, f'lensing/nG_evolution.npy')}")
redshift_list = np.load(f"{os.path.join(root_local, f'lensing/redshift_mid_list.npy')}")

Nbody_end_redshift_list = []
for counter in range(len(boxsize_list)):
	intervals = np.load(f"{os.path.join(root_local, f'shells/shell_intervals_b{len(boxsize_list)-counter}.npy')}")
	Nbody_end_redshift_list.append(results.redshift_at_comoving_radial_distance(intervals[0][1]/(ql_param['H0']/100)))

mask = redshift_list < 10
filtered_redshift = redshift_list[mask]
params_list = [nG_array[mask, i] for i in range(9)]
param_names = [r'$\sigma$', r'$\sigma_1$', r'$S^{(0)}$', r'$S^{(1)}$', r'$S^{(2)}$'
, r'$K^{(0)}$', r'$K^{(1)}$', r'$K_{1}^{(2)}$', r'$K_{2}^{(2)}$']

fig, axs = plt.subplots(3, 3, figsize=(8, 8))
for i, ax in enumerate(axs.flat):
	ax.axvline(x=Nbody_end_redshift_list[0], color='red',linestyle='--')
	for counter in range(len(boxsize_list)-1):
		ax.axvline(x=Nbody_end_redshift_list[counter+1], color='gray',linestyle='--')
	ax.semilogx(filtered_redshift, params_list[i], label=param_names[i])
	ax.set(title=param_names[i], xlabel='source redshift')
plt.tight_layout()
plt.savefig(os.path.join(root_local, f'lensing/nG_evolution_in_order.png'))
plt.savefig(os.path.join(root_local, f'lensing/pdf/nG_evolution_in_order.pdf'))
plt.close()

