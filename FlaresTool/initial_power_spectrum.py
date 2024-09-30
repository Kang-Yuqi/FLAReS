import camb
import numpy as np

def Pk(z,ql_param,nstep=800,kmax=1e3,h_unit=True, k_hunit=True,nonlinear=True):

    ######################## cosmology with CAMB ##################
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=ql_param['H0'], ombh2=ql_param['ombh2'], omch2=ql_param['omch2'], 
        tau=ql_param['tau'], mnu=ql_param['mnu'], omk=ql_param['omk'])
    pars.InitPower.set_params(As=ql_param['As'], ns=ql_param['ns'])

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