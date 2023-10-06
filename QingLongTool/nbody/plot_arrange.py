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