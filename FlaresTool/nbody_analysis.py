import MAS_library as MASL
import Pk_library as PKL
import numpy as np
import readgadget

def compute_power_spectrum(box_base_name, grid=512, MAS='CIC', ptype=[1],verbose=False, axis=0, threads=2):

    header   = readgadget.header(box_base_name)
    BoxSize  = header.boxsize
    redshift = header.redshift

    pos = readgadget.read_block(box_base_name, "POS ", ptype)

    # Initialize delta grid
    delta = np.zeros((grid, grid, grid), dtype=np.float32)
    
    # Compute the density contrast field using the specified mass-assignment scheme
    MASL.MA(pos.astype(np.float32), delta, BoxSize, MAS, verbose=verbose)
    
    # Normalize the density contrast field
    delta /= np.mean(delta, dtype=np.float64)
    delta -= 1.0
    
    # Calculate the power spectrum using Pk_library
    Pk = PKL.Pk(delta, BoxSize, axis, MAS, threads, verbose)
    
    # Extract the resulting k and monopole (Pk0) from Pk object
    k = Pk.k3D
    Pk0 = Pk.Pk[:, 0] #monopole
    
    return k, Pk0, redshift