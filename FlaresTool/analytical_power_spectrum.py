import camb
import numpy as np
from astropy.constants import c

def Cl_matter(z_list, pars, lmin=2, lmax=1024, kmax=10, nonlinear=True,hubble_units=False,k_hunit=False):
    results = camb.get_results(pars)
    H0 = pars.H0

    PK = camb.get_matter_power_interpolator(pars, zs=z_list, nonlinear=nonlinear, hubble_units=hubble_units, 
        k_hunit=k_hunit, kmax=kmax, var1=None, var2=None)
    
    output = []
    for z in z_list:
        ls = np.arange(lmax+1, dtype=np.float64)
        cl_matter = np.zeros(ls.shape)
        chi = results.comoving_radial_distance(z, tol=0.0001)
        ks = (ls + 0.5) / chi

        cl_matter = PK.P(z, ks, grid=False)
        cl_matter[:lmin] = 0
        output.append([ls,ks,cl_matter])

    return output


def lensing_kappa_CAMB(pars, lmax=2500, lens_potential_accuracy=4):

    pars.set_for_lmax(lmax, lens_potential_accuracy=lens_potential_accuracy)
    results = camb.get_results(pars)
    cl_camb = results.get_lens_potential_cls(lmax=lmax)

    return [np.arange(lmax+1),cl_camb[:, 0]/4*(2*np.pi)] # [\ell, [\ell(\ell+1)]^2C_\ell^{\phi\phi}/4]


def lensing_kappa_Cl_range(pars,num_split=100,lmin=0,lmax = 2500,z_low=None,z_high=None):

    H0 = pars.H0
    h = H0/100
    results = camb.get_results(pars)
    chistar = (results.conformal_time(0) - results.tau_maxvis)

    omde = results.get_Omega('de', z=0)
    omc = results.get_Omega('cdm', z=0)
    omb = results.get_Omega('baryon', z=0)
    omrad = results.get_Omega('photon', z=0)
    omneutrino = results.get_Omega('neutrino', z=0)
    omnu = results.get_Omega('nu', z=0)
    om = omc+omb+omrad+omneutrino+omnu

    if z_low is not None and z_high is not None:
        chi_low = results.comoving_radial_distance(z_low, tol=0.0001)
        chi_high = results.comoving_radial_distance(z_high, tol=0.0001)
    elif z_low is not None:
        chi_low = results.comoving_radial_distance(z_low, tol=0.0001)
        chi_high = chistar
    elif z_high is not None:
        chi_high = results.comoving_radial_distance(z_high, tol=0.0001)
        chi_low = 0
    else:
        chi_low = 0
        chi_high = chistar

    split = np.linspace(chi_low,chi_high,num_split+1)
    midpoints = split[1::2][::-1]
    intervals = np.column_stack([split[::2][:-1], split[::2][1:]])[::-1]
    redshifts = results.redshift_at_comoving_radial_distance(np.asarray(midpoints))

    cl_matter_nonlinear_list = Cl_matter(redshifts,pars,lmin=lmin,lmax=lmax,kmax=1e2,hubble_units=False,nonlinear=True)

    cl_dens = np.zeros(np.shape(cl_matter_nonlinear_list[0][2]))
    for i in range(len(redshifts)):
        interval = intervals[i]
        chi = midpoints[i]
        [ls,ks,cl_matter_nonlinear] = cl_matter_nonlinear_list[i]
        cl_dens += cl_matter_nonlinear*((chistar-chi)/chistar)**2/(1/(1+redshifts[i]))**2*(interval[1]-interval[0])
    cl_kappa = cl_dens*(3*H0**2*om/(2*c.to('km/s').value**2))**2

    return [ls,cl_kappa]
