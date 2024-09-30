import numpy as np
import mtneedlet as nd

def needlet_alm(alm, j, B=2,lmax =None):
    import healpy as hp
    bl = nd.standardneedlet(B, j, lmax)
    filtered_alm = hp.almxfl(alm, bl)
    return bl,filtered_alm

def needlet_window_plot(js, B=2, lmax=None, colors_map=None, savepath=None, show=False):
    import matplotlib.pyplot as plt

    ls = np.arange(lmax + 1)
    for i, j in enumerate(js):
        needlet_window = nd.standardneedlet(B, j, lmax)
        color = colors_map[i] if colors_map is not None and i < len(colors_map) else None
        plt.semilogx(ls, needlet_window, label=r"$j=%s$" % j, color=color)
    
    plt.xlabel(r"$\ell$")
    plt.legend()
    plt.grid()
    plt.title(f'standard needlet windows B={B}')
    
    if savepath is not None:
        plt.savefig(f'{savepath}')
    if show:
        plt.show()
    else:
        plt.close()
