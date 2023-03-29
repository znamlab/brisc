"""Small script to threshold downsampled data and find cells in injection site"""
import pathlib

import numpy as np
from skimage import measure
from matplotlib import pyplot as plt
from cricksaw_analysis.main import find_cell_in_injection_site
import flexiznam as flz

NAPARI = False
project = 'rabies_barcoding'

mouse = 'BRYC64.2h'
channel = 3
pixel_size_um = 50
data_size_um = [8, 2, 2]

out = find_cell_in_injection_site(project, mouse, channel, pixel_size_um,
                                  cell_scale_um=data_size_um, inj_min_width=100,
                                  inj_vol_percentile=99.99, cell_type='Cells',
                                  cell_kwargs=None, view_in_napari=NAPARI)
cell_in_inj, inj_site, cell_out_inj = out[:3]

inj_center = measure.centroid(inj_site) * pixel_size_um
cells = np.vstack([cell_in_inj, cell_out_inj]) * np.array(data_size_um)
dst_to_center = np.sqrt(np.sum((cells - inj_center) ** 2, axis=1))
dst_to_center = np.sort(dst_to_center) / 1000
fig, axes = plt.subplots(2, 2)
axes[0, 0].hist(dst_to_center, np.arange(0, 10, 0.1))
axes[0, 1].plot(dst_to_center, np.arange(len(dst_to_center)))
axes[1, 0].plot(dst_to_center, np.arange(len(dst_to_center)) / len(dst_to_center))
axes[1, 1].plot(dst_to_center, np.arange(len(dst_to_center)))

for ax in axes.flatten():
    ax.set_xlabel('Distance to injection (mm)')
for ax in axes[1, :]:
    ax.set_xlim(0, 2)
axes[0, 0].set_ylabel('Number of cells')
axes[0, 1].set_ylabel('Cumulative number of cells')
axes[1, 1].set_ylabel('Cumulative number of cells')
axes[1, 0].set_ylabel('Cumulative fraction of cells')
axes[1, 0].set_ylim(0, 0.5)
axes[1, 1].set_ylim(0, 10000)
fig.suptitle(mouse)
fig.subplots_adjust(wspace=0.5, hspace=0.4)
fig.savefig(pathlib.Path(flz.PARAMETERS['data_root']['processed']) / project / mouse /
            'cell_vs_distance.svg', dpi=600)
fig.savefig(pathlib.Path(flz.PARAMETERS['data_root']['processed']) / project / mouse /
            'cell_vs_distance.pdf', dpi=600)
plt.show()
voxel_vol = (pixel_size_um * 1e-3) ** 3
inj_vol = np.sum(inj_site) * voxel_vol

nc = len(cell_in_inj) + len(cell_out_inj)
dst_threshold = 1
cells_local = dst_to_center[dst_to_center < dst_threshold]
print('Analysis for %s' % mouse)
print('There are %d cells in total.' % nc)
msg = 'There are %d cells in the injection site.' % len(cell_in_inj)
msg += "That's %.2f %%" % (len(cell_in_inj) / nc * 100)
print(msg)
print('The injection site is %.2f mm^3.' % inj_vol)
print('The density of cells is therefore %d cells/mm^3' % (len(cell_in_inj) / inj_vol))
print('If we assume that there are 100,000 cells/mm^3, we stained %.3f %% of the cells'
      % (len(cell_in_inj) / inj_vol / 100000))
msg = 'There are %d cells < %.1f mm from the injection site.' % (len(cells_local),
                                                                 dst_threshold)
msg += "That's %.2f %%" % (len(cells_local) / nc * 100)
print(msg)

print('Done')

if NAPARI:
    viewer = out[-1]
