import camb
import numpy as np

def Pk(z,nstep=800,kmax=1e3,h_unit=True, k_hunit=True,nonlinear=True,
	H0=None, ombh2=None, omch2=None, mnu=None, omk=None, tau=None,As=None, ns=None):

	pars = camb.CAMBparams()
	pars.set_cosmology(H0=H0,ombh2=ombh2,omch2=omch2,tau=tau,mnu=mnu,omk=omk)
	pars.InitPower.set_params(As=As,ns=ns)
	results = camb.get_results(pars)
	PK = camb.get_matter_power_interpolator(pars,zs=[z], nonlinear=nonlinear,hubble_units=h_unit, k_hunit=k_hunit, 
		kmax=kmax,var1=None,var2=None)

	k=np.exp(np.log(10)*np.linspace(-4,np.log10(kmax),nstep))
	pk_1 = PK.P(z,k)
	pk_2 = np.log10(4*np.pi*k**3*pk_1)

	pk_camb = np.empty((nstep,2))
	pk_camb[:,0] = k
	pk_camb[:,1] = pk_1

	pk_ngenic = np.empty((nstep,2))
	pk_ngenic[:,0] = np.log10(k)
	pk_ngenic[:,1] = pk_2

	return pk_camb,pk_ngenic


## test
# import QingLongTool as ql
# import matplotlib
# import matplotlib.pyplot as plt

# root_local = '/mnt/c/Users/KYQ/'
# root = root_local + 'OneDrive - UNSW/data&code/QingLong/example/gadget4_128cub_4box/'

# param_path = root + 'ql_param.ini'
# ql_param = ql.read_config(param_path)

# pk_camb,pk_ngenic = initial_power_spectrum(ql_param['z_start'],nstep=800,kmax=1e3,h_unit=False, k_hunit=False,nonlinear=True,
# 	H0=ql_param['H0'], ombh2=ql_param['ombh2'], omch2=ql_param['omch2'], 
# 	tau=ql_param['tau'], mnu=ql_param['mnu'], omk=ql_param['omk'], As=ql_param['As'], 
# 	ns=ql_param['ns'])

# plt.loglog(pk_camb[:,0],pk_camb[:,1])
# plt.ylabel(r'$\mathrm{P}(k)~[(\mathrm{Mpc}/h)^3]$')
# plt.xlabel(r'$k[h/\mathrm{Mpc}]$')
# plt.title(r'CAMB ($z_{start}$ = %s)' %ql_param['z_start'])
# plt.show()

# plt.plot(pk_ngenic[:,0],pk_ngenic[:,1])
# plt.ylabel(r'$\log _{10}([4\pi k^3 \mathrm{P}(k)]^2)~[(\mathrm{Mpc}/h)^3]$')
# plt.xlabel(r'$\log _{10}(k)[h/\mathrm{Mpc}]$')
# plt.title(r'N-GenIC input ($z_{start}$ = %s)' %ql_param['z_start'])
# plt.show()
