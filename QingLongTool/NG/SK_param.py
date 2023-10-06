import healpy as hp
import numpy as np


def Ursell_function(x1,x2,x3,x4):

    term1 = np.mean(x1*x2*x3*x4)
    term2 = np.mean(x1)*np.mean(x2*x3*x4)+np.mean(x2)*np.mean(x1*x3*x4)+np.mean(x3)*np.mean(x1*x2*x4)+np.mean(x4)*np.mean(x1*x2*x3)
    term3 = np.mean(x1*x2)*np.mean(x3*x4)+np.mean(x1*x3)*np.mean(x2*x4)+np.mean(x1*x4)*np.mean(x2*x3)
    term4 = np.mean(x1*x2)*np.mean(x3)*np.mean(x4)+np.mean(x1*x3)*np.mean(x2)*np.mean(x4)+np.mean(x1*x4)*np.mean(x2)*np.mean(x3)+np.mean(x2*x3)*np.mean(x1)*np.mean(x4)+np.mean(x2*x4)*np.mean(x1)*np.mean(x3)+np.mean(x3*x4)*np.mean(x1)*np.mean(x2)
    term5 = np.mean(x1)*np.mean(x2)*np.mean(x3)*np.mean(x4)
    result = term1-term2-term3+2*term4-6*term5

    return result

def Laplacian(alm,nside,lmax):
    ls, _ = hp.Alm.getlm(lmax=lmax)
    alm_Laplacian = -(ls * (ls + 1)) * alm
    map_Laplacian = hp.alm2map(alm_Laplacian, nside=nside) 
    return map_Laplacian

def Gradient(alm,nside,lmax):
    [imap,map_dt,map_dp] = hp.sphtfunc.alm2map_der1(alm,nside)
    Gradient_map = np.array([map_dt,map_dp])
    Gradient_map_square = (map_dt)**2 + (map_dp)**2
    return Gradient_map,Gradient_map_square

def S_K_parameter(alm,nside,lmax,mask_file=None,local=False):

    map_input = hp.sphtfunc.alm2map(alm,nside)

    Laplacian_map = Laplacian(alm,nside,lmax)
    _,Gradient_map_square = Gradient(alm,nside,lmax)
    Gradient_mod = np.sqrt(Gradient_map_square)

    fsky = 1
    if mask_file is not None:
        mask_oringnal = hp.read_map(mask_file).astype(np.bool_)
        mask = hp.ud_grade(mask_oringnal, nside)
        n_unmasked = np.count_nonzero(mask)
        map_input = hp.ma(map_input)
        map_input.mask = np.logical_not(mask)
        Laplacian_map = hp.ma(Laplacian_map)
        Laplacian_map.mask = np.logical_not(mask)
        Gradient_map_square = hp.ma(Gradient_map_square)
        Gradient_map_square.mask = np.logical_not(mask)
        Gradient_mod = hp.ma(Gradient_mod)
        Gradient_mod.mask = np.logical_not(mask)

    sigma = np.sqrt(np.mean(map_input*map_input))
    sigma_1 = np.sqrt(-np.mean(map_input*Laplacian_map))

    S = np.mean(map_input*map_input*map_input)/sigma**4
    S1 = (3./2.) * np.mean(map_input*Gradient_map_square)/(sigma**2*sigma_1**2)
    S2 = -3*np.mean(Gradient_map_square*Laplacian_map)/sigma_1**4

    K = Ursell_function(map_input,map_input,map_input,map_input)/sigma**6
    K1 = 2*(np.mean(map_input*map_input*Gradient_map_square)-sigma**2*sigma_1**2)/(sigma**4*sigma_1**2)
    K12 = -(4*(np.mean(map_input*Gradient_map_square*Laplacian_map)+sigma_1**4)+np.mean(Gradient_map_square*Gradient_map_square)-2*sigma_1**4)/(sigma**2*sigma_1**4)
    K22 = -(4*(np.mean(map_input*Gradient_map_square*Laplacian_map)+sigma_1**4)+2*(np.mean(Gradient_map_square*Gradient_map_square)-2*sigma_1**4))/(sigma**2*sigma_1**4)

    if local:
        sigma_sq_map = map_input*map_input
        sigma_1_sq_map = -map_input*Laplacian_map

        S_map = map_input*map_input*map_input/sigma**4
        S1_map = (3./2.) * map_input*Gradient_map_square/(sigma**2*sigma_1**2)
        S2_map = -3*Gradient_map_square*Laplacian_map/sigma_1**4

        K_map = ((map_input*map_input*map_input*map_input)-3*sigma**4)/sigma**6
        K1_map = 2*(map_input*map_input*Gradient_map_square-sigma**2*sigma_1**2)/(sigma**4*sigma_1**2)
        K12_map = -(4*(map_input*Gradient_map_square*Laplacian_map+sigma_1**4)+Gradient_map_square*Gradient_map_square-2*sigma_1**4)/(sigma**2*sigma_1**4)
        K22_map = -(4*(map_input*Gradient_map_square*Laplacian_map+sigma_1**4)+2*(Gradient_map_square*Gradient_map_square-2*sigma_1**4))/(sigma**2*sigma_1**4)

        return [sigma,sigma_1,S,S1,S2,K,K1,K12,K22,sigma_sq_map,sigma_1_sq_map,S_map,S1_map,S2_map,K_map,K1_map,K12_map,K22_map]
    else:
        return [sigma,sigma_1,S,S1,S2,K,K1,K12,K22]