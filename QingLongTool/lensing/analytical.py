import camb
from camb import model, initialpower
import numpy as np
import matplotlib.pyplot as plt

def Cl_matter(z_list, lmin=2, lmax=1024, kmax=1e2, nonlinear=True, 
    H0=None, ombh2=None, omch2=None, mnu=None, omk=None, tau=None, As=None, ns=None):

    pars = camb.CAMBparams()
    pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, tau=tau, mnu=mnu, omk=omk)
    pars.InitPower.set_params(As=As, ns=ns)
    results = camb.get_results(pars)

    PK = camb.get_matter_power_interpolator(pars, zs=z_list, nonlinear=nonlinear, hubble_units=True, 
        k_hunit=True, kmax=kmax, var1=None, var2=None)
    
    output = []
    for z in z_list:
        ls = np.arange(lmin, lmax+1, dtype=np.float64)
        cl_matter = np.zeros(ls.shape)
        chi = results.comoving_radial_distance(z, tol=0.0001) * H0 / 100 # in Mpc/h
        ks = []
        for i, l in enumerate(ls):
            k = (l + 0.5) / chi
            cl_matter[i] = PK.P(z, k, grid=False)
            ks.append(k)
        output.append([ls,np.array(ks),cl_matter])

    return output

def lensing_kappa_CAMB(lmax=1024,H0=None, ombh2=None, omch2=None, mnu=None, omk=None, tau=None,As=None, ns=None):

    h = H0/100
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=H0,ombh2=ombh2,omch2=omch2,tau=tau,mnu=mnu,omk=omk)
    pars.InitPower.set_params(As=As,ns=ns)
    pars.set_for_lmax(lmax, lens_potential_accuracy=1)
    results = camb.get_results(pars)
    cl_camb = results.get_lens_potential_cls(lmax=lmax)
    return [np.arange(lmax+1),cl_camb[:, 0]/4*(2*np.pi)]

