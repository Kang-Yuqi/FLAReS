import numpy as np
from scipy.integrate import quad


def integrand(k_parallel, ell,z,PK,Delta_r,r):
    k_perp = (ell) / r
    k = np.sqrt(k_parallel**2 + k_perp**2)
    return PK.P(z, k, grid=False) * (np.sin(k_parallel * Delta_r /2)/(k_parallel * Delta_r/2 ))**2 /np.pi # a factor of 2 is added, 
																										  #   consider k_parallel belong to [-inf,inf]
def C_l_shell(ell,z,PK,Delta_r,r,k_parallel_min,k_parallel_max):
    integral, error = quad(integrand, k_parallel_min, k_parallel_max, args=(ell,z,PK,Delta_r,r), limit=200) 
    return integral

def compute_spectrum(lmin,lmax,z,PK,Delta_r,r,k_parallel_min,k_parallel_max):
    Cls = []
    for ell in range(0,lmin):
    	Cls.append(0)
    for ell in range(lmin, lmax + 1):
        Cl = C_l_shell(ell,z,PK,Delta_r,r,k_parallel_min,k_parallel_max)
        Cls.append(Cl)
    return np.array(Cls)
