"""Look at on brain and find where are the presynaptic cells in the ARA"""
import matplotlib

matplotlib.use('macosx')
from pathlib import Path
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from cricksaw_analysis.io import load_cellfinder_results
import flexiznam as flz
import bg_atlasapi as bga
from cricksaw_analysis import atlas_utils
import pandas as pd

NAPARI = False
project = 'rabies_barcoding'
REDO = False

mice = ['BRYC64.2h', 'BRYC64.2i']
channel = 3
data_size_um = [8, 2, 2]
atlas_size = 25

processed = Path(flz.PARAMETERS['data_root']['processed'])
raw = Path(flz.PARAMETERS['data_root']['raw'])

bg_atlas = bga.bg_atlas.BrainGlobeAtlas('allen_mouse_%dum' % atlas_size)
cdf = atlas_utils.create_ctx_table(bg_atlas)

count_by_mouse = dict()
for mouse in mice:
    mouse_cellfinder_folder = processed / project / mouse / 'cellfinder_results_050'
    summary_density_mouse_path = processed / project / mouse / ('%s_summary_density.csv' %
                                                                mouse)

    if (not REDO) and summary_density_mouse_path.is_file():
        # mouse already analysed. Read data from disk
        count_by_mouse[mouse] = pd.read_csv(summary_density_mouse_path, index_col=0)
    else:
        cells, downsampled_stacks, atlas = load_cellfinder_results(mouse_cellfinder_folder)

        rd = np.array(np.round(cells.values), dtype=int)
        atlas_id = atlas[rd[:, 0], rd[:, 1], rd[:, 2]]

        cell_count = atlas_utils.cell_density_by_areas(atlas=atlas, atlas_id=atlas_id,
                                                       bg_atlas=bg_atlas, cortex_df=cdf,
                                                       pixel_size=atlas_size)
        cell_count = pd.DataFrame(cell_count)
        cell_count.to_csv(summary_density_mouse_path)
        count_by_mouse[mouse] = cell_count

# now do stuff

long_df = []
for mouse, df in count_by_mouse.items():
    df = df.T
    df['mouse'] = mouse
    n_cell = df['count'].sum()
    df['proportion'] = df['count'] / n_cell
    long_df.append(df)
long_df = pd.concat(long_df)
long_df = long_df.reset_index().rename(columns=dict(index='area'))


ctx_area = cdf.area_acronym.unique()
av = long_df.groupby('area').aggregate(np.mean).sort_values('proportion')

fig = plt.figure()

fig.clear()
ax = fig.add_subplot(2, 2, 1)
sns.stripplot(ax=ax, data=long_df, x='area', y='count', hue='mouse',
              order=av.index[-35:])
ax.set_xticklabels([])
ax = fig.add_subplot(2, 2, 3)
sns.stripplot(ax=ax, data=long_df, x='area', y='count', hue='mouse',
              order=av.index[-15:])
ax.set_xticklabels([bg_atlas.structures[l._text]['name'] for l in ax.get_xticklabels()],
                   rotation=90)
ax.set_ylim([0, av.iloc[-3]['count']])
ax.get_legend().remove()

# same for density
av_d = 0
ax = fig.add_subplot(2, 2, 2)
sns.stripplot(ax=ax, data=long_df, x='area', y='density', hue='mouse',
              order=av.index[-15:])
ax.set_xticklabels([])
ax.get_legend().remove()
ax = fig.add_subplot(2, 2, 4)
sns.stripplot(ax=ax, data=long_df, x='area', y='density', hue='mouse',
              order=av.index[-15:])
ax.set_ylim([0, av.iloc[-3]['density']])
ax.set_xticklabels([bg_atlas.structures[l._text]['name'] for l in ax.get_xticklabels()],
                   rotation=90)
ax.get_legend().remove()
for x in fig.axes:
    x.set_xlabel('')
fig.subplots_adjust(top=0.98, right=0.95, hspace=0.05, wspace=0.3, bottom=0.4)
fig.savefig(processed / project / 'cell_count_summary.png', dpi=300)