import numpy as np
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
from matplotlib import patches

def create_ax(fig,max_len,position=[0,0.15,1,0.8]):
    ax = fig.add_axes(position)
    ax.set_aspect('equal')
    ax.set_xlim([0, max_len])
    ax.set_ylim([0, max_len])
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    ax.set_frame_on(False)
    return ax

def draw_rotated_grid_circle(ax, grid_size=None, radius=None,intervals=None, rotation_angle=0,colors = ['yellow', 'blue']):

    max_extent = radius + grid_size
    
    t = transforms.Affine2D().rotate_deg(rotation_angle) + ax.transData

    # Create a grid filled with the grid pattern
    x = np.arange(-max_extent, max_extent, grid_size)
    y = np.arange(-max_extent, max_extent, grid_size)
    for i in x:
        ax.plot([i, i], [y[0], y[-1]], color='k', linewidth=0.5, transform=t)
    for i in y:
        ax.plot([x[0], x[-1]], [i, i], color='k', linewidth=0.5, transform=t)

    circle_patch = patches.Circle((0, 0), radius, facecolor='none')
    ax.add_patch(circle_patch)

    for i in range(len(intervals)):
        annulus = patches.Annulus((0, 0), intervals[i][1], intervals[i][1]-intervals[i][0]-1e-4, facecolor=colors[i % 2], alpha=0.5)
        ax.add_patch(annulus)
        
    for line in ax.lines:
        line.set_clip_path(circle_patch)

def draw_rotated_grid_annulus(ax, grid_size=None, radius=None,width=None,intervals=None, rotation_angle=0,colors = ['yellow', 'blue']):

    max_extent = radius + grid_size
    
    t = transforms.Affine2D().rotate_deg(rotation_angle) + ax.transData

    # Create a grid filled with the grid pattern
    x = np.arange(-max_extent, max_extent, grid_size)
    y = np.arange(-max_extent, max_extent, grid_size)
    for i in x:
        ax.plot([i, i], [y[0], y[-1]], color='k', linewidth=0.5, transform=t)
    for i in y:
        ax.plot([x[0], x[-1]], [i, i], color='k', linewidth=0.5, transform=t)

    annulus_patch = patches.Annulus((0, 0), radius,width, facecolor='none')
    ax.add_patch(annulus_patch)

    for i in range(len(intervals)):
        annulus = patches.Annulus((0, 0), intervals[i][1], intervals[i][1]-intervals[i][0]-1e-4, facecolor=colors[i % 2], alpha=0.5)
        ax.add_patch(annulus)
    
    for line in ax.lines:
        line.set_clip_path(annulus_patch)


def plot_shell_arrange(fl_param, root_local):
    import camb
    import os
    num_shells = fl_param['num_shells']
    n = sum(num_shells)
    N = fl_param['Gaussian_shell_steps']
    ######################## cosmology with CAMB ##################
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=fl_param['H0'], ombh2=fl_param['ombh2'], omch2=fl_param['omch2'], 
        tau=fl_param['tau'], mnu=fl_param['mnu'], omk=fl_param['omk'])
    pars.InitPower.set_params(As=fl_param['As'], ns=fl_param['ns'])
    results = camb.get_results(pars)
    chistar = (results.conformal_time(0) - results.tau_maxvis)*fl_param['H0']/100

    boxsize_list = np.loadtxt(os.path.join(root_local, f"{fl_param['Nbody_folder']}/boxsize_list.txt"))
    num_box = len(boxsize_list)

    intervals_list = []
    for idx in range(num_box):
        intervals = np.load(os.path.join(root_local, f"shells/shell_intervals_{n}Boxshells_b{idx+1}.npy"))[::-1]
        intervals_list.append(intervals)

    intervals_G = np.load(os.path.join(root_local, f'shells/shell_intervals_{n}Boxshells_{N}Gshell_G.npy'))[::-1]
    intervals_list.append(intervals_G)
    boxsize_list = np.append(boxsize_list,1e6)

    num_box = num_box+1

    max_radi = intervals_list[num_box-1][-1][1]
    fig = plt.figure(figsize=(6, 6))
    axes = [create_ax(fig,max_radi) for _ in range(num_box)]

    for idx in range(num_box):
        if idx == 0:
            draw_func = draw_rotated_grid_circle
            kwargs = {
                "rotation_angle": 45,
                "colors": ['yellow', 'blue']
            }
        else:
            draw_func = draw_rotated_grid_annulus
            kwargs = {
                "width": intervals_list[idx][-1][1] - intervals_list[idx][0][0],
                "rotation_angle": 0 if idx % 2 == 1 else 45,
                "colors": [ 'lightgrey','limegreen'] if idx % 2 == 1 else ['yellow', 'blue']
            }

        draw_func(axes[idx],
                  grid_size=boxsize_list[idx],
                  radius=intervals_list[idx][-1][1],
                  intervals=intervals_list[idx],
                  **kwargs)


    tick_values = [0] + [interval[-1][1] for interval in intervals_list]
    axes[num_box-1].xaxis.set_visible(True)
    axes[num_box-1].set_xticks(tick_values)
    axes[num_box-1].set_xlabel('comoving distance [Mpc/h]')
    axes[num_box-1].xaxis.set_label_coords(0.5, -0.1)

    plt.savefig(os.path.join(root_local, f"{fl_param['Nbody_folder']}/nbody_arrange.png"))
    plt.savefig(os.path.join(root_local, f"{fl_param['Nbody_folder']}/nbody_arrange.pdf"))

    ## relabel with redshift
    tick_values_label = np.round(results.redshift_at_comoving_radial_distance(np.array(tick_values) / (fl_param['H0'] / 100)), 2)
    axes[num_box-1].set_xticklabels(tick_values_label, rotation=30)
    axes[num_box-1].set_xlabel('redshift')

    plt.savefig(os.path.join(root_local, f"{fl_param['Nbody_folder']}/nbody_arrange_redshift.png"))
    plt.close()