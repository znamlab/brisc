"""Script to look at the density of PhP.eB cre cells"""
import matplotlib
import socket
if socket.gethostname() == 'C02Z85AULVDC':
    matplotlib.use('macosx')
import seaborn as sns
from pathlib import Path
import numpy as np
import pandas as pd
import bg_atlasapi as bga
from matplotlib import pyplot as plt
from sklearn.neighbors import KDTree
import flexiznam as flz
from cricksaw_analysis import atlas_utils
from cricksaw_analysis.io import load_cellfinder_results
from cricksaw_analysis.atlas_utils import cell_density_by_areas

REDO = False
NAPARI = False
PLOT_DENSITY = False
PLOT_SUMMARY = True
project = 'rabies_barcoding'
virus = 'A87'
# brain_pixel_size = np.array([8, 2, 2], dtype=float)
atlas_pixel_size = 25.
cell_type = 'Cells'
channel = 0
cortical_areas = 'ALL'


processed = Path(flz.PARAMETERS['data_root']['processed'])
raw = Path(flz.PARAMETERS['data_root']['raw'])

mouse_csv = processed / project / 'mice_list.csv'

mice_df = pd.read_csv(mouse_csv, skipinitialspace=True)
mice_df.set_index('Mouse', inplace=True, drop=False)

bg_atlas = bga.bg_atlas.BrainGlobeAtlas('allen_mouse_25um')
cdf = atlas_utils.create_ctx_table(bg_atlas)
fig = plt.figure(figsize=(10, 5))

if cortical_areas == 'ALL':
    cortical_areas = list(cdf.area_acronym.unique())


def calculate_neighbourhood(distance_radii, kdtree, cells, cdf=cdf, bylayer=False,
                            area_acronym='VISp'):
    neighbours = dict()
    distance = dict()
    if bylayer:
        layers =  ['all', '1', '2/3', '4', '5', '6a', '6b']
    else:
        layers = ['all']
    for layer in layers:
        v1_indices = cdf.loc[cdf.area_acronym == area_acronym, :]
        if layer == 'all':
            v1_indices = v1_indices.id
        else:
            v1_indices = v1_indices.loc[v1_indices.layer == layer, 'id']
        cells_in_layer = np.isin(atlas_id, v1_indices)
        vlc_um = cells[cells_in_layer] * atlas_pixel_size
        dist, neigh = ctx_tree.query(vlc_um, k=2, return_distance=True)
        # the closest neighbour is always yourself
        distance[layer] = dist[:, 1]

        neighbours[layer] = np.vstack([kdtree.query_radius(vlc_um, r, count_only=True)
                                       for r in distance_radii]) - 1
    return neighbours, distance


summary_density = dict()
filter_indices = cdf.id
area = 'VISp'

for mouse, m_df in mice_df[mice_df['Virus Batch'] == 'A87'].iterrows():
    if mouse in []:
        continue
    mouse_cellfinder_folder = processed / project / mouse / 'cellfinder_results'
    if not mouse_cellfinder_folder.is_dir():
        print('No cellfinder folder for %s' % mouse)
        print(mouse_cellfinder_folder)
        continue
    print('Doing %s' % mouse)
    density_fig_path = processed / project / mouse / (
            '%s_neighbour_by_distance.png' % mouse)
    summary_density_mouse_path = processed / project / mouse / (
            '%s_summary_density.csv' % mouse)

    need_data = False
    if REDO:
        need_data = True
    if PLOT_DENSITY and (not density_fig_path.is_file()):
        need_data = True
    if PLOT_SUMMARY and (not summary_density_mouse_path.is_file()):
        need_data = True

    if need_data:
        try:
            cells, downsampled_stacks, atlas = load_cellfinder_results(
                                                mouse_cellfinder_folder)
        except IOError or FileNotFoundError as err:
            print('Failed to load data: %s' % err)
            continue
        rd = np.array(np.round(cells.values), dtype=int)
        atlas_id = atlas[rd[:, 0], rd[:, 1], rd[:, 2]]
    if PLOT_SUMMARY:
        if REDO or (not summary_density_mouse_path.is_file()):
            assert need_data
            summary_density[mouse] = pd.DataFrame(cell_density_by_areas(atlas_id,
                                                                        atlas))
            summary_density[mouse].to_csv(summary_density_mouse_path)
        else:
            summary_density[mouse] = pd.read_csv(summary_density_mouse_path,
                                                 index_col=0)

    if PLOT_DENSITY:
        # keep only cortical cells
        v1_indices = cdf.loc[cdf.area_acronym == area, 'id']

        cell_in_v1 = np.isin(atlas_id, v1_indices)
        cell_in_ctx = np.isin(atlas_id, filter_indices)
        ctx_um = cells[cell_in_ctx] * atlas_pixel_size
        ctx_tree = KDTree(ctx_um)

        distance_radii = np.arange(0, 1000, 10)
        volume = 4/3 * np.pi * (distance_radii/1e3)**3
        neighbours, distance = calculate_neighbourhood(distance_radii,
                                                       kdtree=ctx_tree,
                                                       cells=cells)
        fig.clear()
        ax0 = fig.add_subplot(2, 3, 1)
        ax0.set_title('Total: %d cells, Cortex %d cells, V1: %d cells'
                      % (len(cells), np.sum(cell_in_ctx), np.sum(cell_in_v1)))
        cutoff = 1000
        n, b, p = ax0.hist(distance['all'], bins=np.arange(0, cutoff, 5))
        ax0.set_xlim([0, b[:-1][n > 0][-1]])
        ax0.set_xlabel('Distance to first neighbour')
        ax0.set_ylabel('Number of cells')

        ax1 = fig.add_subplot(2, 3, 2)
        rad = 100
        radinx = distance_radii.searchsorted(rad)
        ax1.hist(neighbours['all'][radinx], bins=np.arange(-0.5, 100.5))
        ax1.set_xlim(-0.5, 100)
        ax1.set_xlabel('# of neighbours in %d um radius' % rad)
        ax1.set_ylabel('# of cells')

        ax2 = fig.add_subplot(2, 3, 3)
        ax2.clear()
        ax2.set_ylabel('Fraction of cells\nwith Ns neighbour')
        ax2.set_xlabel('Distance')
        for n in [1, 5, 10, 20, 50]:
            has_neigh = neighbours['all'] >= n
            frac = np.sum(has_neigh, axis=1) / len(has_neigh[0])
            ax2.plot(distance_radii, frac, label=str(n))
        ax2.set_xlim([0, distance_radii[frac.argmax()]])
        ax2.set_ylim([0, 1])
        ax2.legend(loc=0)

        ax3 = fig.add_subplot(2, 2, 3)
        ax4 = fig.add_subplot(2, 2, 4)
        ax3.set_xlabel('Distance')
        ax3.set_ylabel('Mean number of neighbours')
        ax4.set_xlabel('Distance')
        ax4.set_ylabel('Mean cell denstity (cells/mm$^3$)')

        cutoff = 500
        beg_density = np.searchsorted(distance_radii, 50)
        for layer, nei in neighbours.items():
            m = np.mean(nei, axis=1)
            std = np.std(nei, axis=1)
            ax3.plot(distance_radii, m, label=layer, lw=3 if layer == 'all' else 1,
                     zorder=3 if layer == 'all' else 1)
            if layer == 'all':
                dens = (nei[beg_density:, :].T / volume[beg_density:]).T
                m = np.mean(dens, axis=1)
                std = np.std(dens, axis=1)
                ax4.plot(distance_radii[beg_density:], m, label=layer, lw=3,
                         zorder=3)
                ax4.fill_between(distance_radii[beg_density:], m-std, m+std,
                                 color='grey', alpha=0.4)
                ax3.set_ylim([0, m[distance_radii.searchsorted(cutoff)]])
        ax3.set_xlim([0, cutoff])
        ax4.set_xlim([0, cutoff])
        ax3.legend(loc='upper center', bbox_to_anchor=(
            0.5, 1.05), ncol=4, fontsize=6)
        fig.subplots_adjust(wspace=0.4, hspace=0.4)
        fig.suptitle('%s - dilution %s' % (mouse, m_df.Dilution))
        fig.savefig(density_fig_path, dpi=600)
        fig.savefig(str(density_fig_path).replace('png', 'svg'), dpi=600)
        if NAPARI:
            from napari.viewer import Viewer
            viewer = Viewer()
            img = downsampled_stacks['downsampled_channel_%d' % channel]
            viewer.add_image(img, name='Channel %d' % channel)
            viewer.add_labels(atlas, name='Registered atlas', opacity=0.1)
            viewer.add_points(cells, name='Cells', face_color='blue', opacity=0.2,
                              size=5)
            viewer.add_points(cells[cell_in_v1], name='V1 Cells',
                              face_color='orange', opacity=0.5, size=5)

if PLOT_SUMMARY:
    # we got data for all mice
    # get area volumes
    area_prop = dict()
    px_volume = (atlas_pixel_size / 1000)**3
    for c, adf in cdf.groupby('area_acronym'):
        area_prop[c] = dict()
        area_prop[c]['n_pixels'] = np.sum(np.isin(bg_atlas.annotation, adf.id))
        area_prop[c]['volume'] = area_prop[c]['n_pixels'] * px_volume

    long_format = []
    for mouse, mdata in summary_density.items():
        for area, adata in mdata.items():
            d = dict(area=area, mouse=mouse, dilution=mice_df.loc[mouse, 'Dilution'],
                     zres=mice_df.loc[mouse, 'Zresolution'])
            for w, value in adata.items():
                d['what'] = w
                d['value'] = value
                long_format.append(dict(d))
    long_format = pd.DataFrame(long_format)

    count_df = summary_density
    prop = pd.DataFrame(area_prop)

    # Area figures
    fig = plt.figure()
    for area in ['PIR', 'VISa', 'VISal', 'VISam', 'VISl', 'VISli', 'VISp', 'VISpl',
                 'VISpm', 'VISpor', 'VISrl']:
        v1 = long_format[long_format.area == area]
        vdf = v1[v1.what == 'volume'].set_index('mouse', inplace=False)
        v1 = v1[v1.what == 'count'].set_index('mouse', inplace=False)
        v1.columns = ['count' if c == 'value' else c for c in v1.columns]
        v1['density'] = v1['count'] / vdf['value']
        fig.clear()
        for iax, w in enumerate(['count', 'density']):
            ax = fig.add_subplot(2, 2, 1 + iax * 2)
            sns.stripplot(data=v1, x='dilution', y=w, ax=ax,
                          hue='zres', order=['1/100', '1/330', '1/1000', '1/3300'])
            ax1 = fig.add_subplot(2, 2, 2 + iax * 2)
            sns.stripplot(data=v1, x='dilution', y=w, ax=ax1,
                          hue='zres', order=['1/100', '1/330', '1/1000', '1/3300'])
            ax1.set_ylim([0, v1[v1['dilution'] == '1/1000'][w].max()])
        fig.subplots_adjust(wspace=0.3, hspace=0.3)
        fig.suptitle(area)
        fig.savefig(processed / project / ('%s_density_plot.png' % area), dpi=600)

    # hypothalamus figure
    fig.clear()
    hypo = ['HY'] + bg_atlas.get_structure_descendants('HY')
    # remove useless area:
    hypo.remove('ZI')
    hypo.remove('STN')
    # keep only valid areas
    hypo_df = long_format[long_format.area.isin(hypo)]
    sum_df = hypo_df.groupby(['what', 'mouse', 'dilution', 'zres'],
                             as_index=False).aggregate(np.sum)
    vdf = sum_df[sum_df.what == 'volume'].set_index('mouse', inplace=False, drop=False)
    sum_df = sum_df[sum_df.what == 'count']
    sum_df.set_index('mouse', inplace=True, drop=False)
    sum_df.drop('what', axis=1, inplace=True)
    sum_df.columns = ['count' if c == 'value' else c for c in sum_df.columns]
    sum_df['density'] = sum_df['count'] / vdf.value

    # plot the hypothalamus figure now
    fig.clear()
    plt.suptitle('Hypothalamus (except ZI and STN)')
    ax = fig.add_subplot(2, 2, 1)
    sns.stripplot(data=sum_df, x='dilution', y='count', ax=ax,
                  hue='zres', order=['1/100', '1/330', '1/1000', '1/3300'])
    ax.set_ylabel('Cell count')

    ax1 = fig.add_subplot(2, 2, 2)
    ax1.clear()
    sns.stripplot(data=sum_df, x='dilution', y='density',
                  ax=ax1, hue='zres', order=['1/100', '1/330', '1/1000', '1/3300'])
    ax1.set_ylabel('Cell density (cell/$mm^3$)')

    ax2 = fig.add_subplot(2, 2, 3)
    one_over_100 = hypo_df[hypo_df.dilution == '1/100']
    count_df = one_over_100[one_over_100.what == 'count']
    count_df = count_df.rename(columns=dict(value='count'))
    av = count_df[count_df.zres == 8].groupby('area').aggregate(np.mean)
    av.sort_values('count', inplace=True)
    av['area'] = av.index
    areas_to_keep = av.index[-15:]
    ax2.clear()
    sns.stripplot(data=count_df[count_df.area.isin(areas_to_keep)], x='area', y='count',
                  ax=ax2, hue='zres', order=areas_to_keep)
    sns.stripplot(data=av[av.area.isin(areas_to_keep)], x='area', y='count',
                  order=areas_to_keep, color='k', marker='s', size=5, ax=ax2)
    ax2.tick_params(axis='x', labelrotation=45)
    ax2.set_ylabel('Cell count')

    c = one_over_100[one_over_100.what == 'count'].set_index(['mouse', 'area'])
    v = one_over_100[one_over_100.what == 'volume'].set_index(['mouse', 'area'])
    v['density'] = c.value / v.value
    density_df = v.reset_index()
    av = density_df[density_df.zres == 8].groupby('area').aggregate(np.mean)
    av.sort_values('density', inplace=True)
    av['area'] = av.index
    areas_to_keep = av.index[-15:]

    ax3 = fig.add_subplot(2, 2, 4)
    ax3.clear()
    sns.stripplot(data=density_df[density_df.area.isin(areas_to_keep)], x='area',
                  y='density', ax=ax3, hue='zres', order=areas_to_keep)
    sns.stripplot(data=av.loc[areas_to_keep], x='area', y='density', order=areas_to_keep,
                  color='k', marker='s', size=5, ax=ax3)
    ax3.tick_params(axis='x', labelrotation=45)
    ax3.set_ylabel('Cell density (cell/$mm^3$)')
    fig.savefig(processed / project / ('%s_density_plot.png' % 'hypothalamus'), dpi=600)

    # global figure
    good_areas = dict()
    areas_of_interest = ['ENT', 'PIR', 'AON', 'COA']
    for parent_area in areas_of_interest:
        area_ids = [parent_area] + bg_atlas.get_structure_descendants(parent_area)
        df = long_format[long_format.area.isin(area_ids)]
        df = df.groupby(['mouse', 'dilution', 'zres', 'what']).aggregate(
            np.sum).reset_index()
        df['area'] = parent_area
        good_areas[parent_area] = df

    good_areas = pd.concat(good_areas.values())

    good_areas = good_areas[good_areas.zres == 8]
    c = good_areas[good_areas.what == 'count'].set_index(['mouse', 'area'])
    v = good_areas[good_areas.what == 'volume'].set_index(['mouse', 'area'])
    c['density'] = c.value / v.value
    good_areas = c.reset_index()
    good_areas.rename(columns=dict(value='count'), inplace=True)
    av = good_areas.groupby(['dilution',  'area']).aggregate(np.mean).reset_index()
    narea = len(good_areas.area.unique())
    ylabel = dict(count='Cell count', density='Cell density ($cell.mm^{-3}$)')
    xlabel = dict(count='', density='Dilution')

    fig, axes = plt.subplots(2, 1)
    mean_only = False
    if mean_only:
        good_areas = good_areas.groupby(['dilution', 'area']).aggregate(
            np.mean).reset_index()
    for iax, what in enumerate(['count', 'density']):
        sns.stripplot(data=good_areas, x='dilution', y=what, hue='area',
                      order=['1/100', '1/330', '1/1000', '1/3300'], dodge=True,
                      ax=axes[iax], hue_order=areas_of_interest, size=5, alpha=0.7,
                      palette='Set1')
        if not mean_only:
            sns.boxplot(data=good_areas, x='dilution', y=what, hue='area',
                        order=['1/100', '1/330', '1/1000', '1/3300'], dodge=True,
                        ax=axes[iax], hue_order=areas_of_interest, showmeans=True,
                        meanline=True, meanprops={'color': 'k', 'ls': '-', 'lw': 2},
                        medianprops={'visible': False}, whiskerprops={'visible': False},
                        showfliers=False, showbox=False, showcaps=False,)
        handles, labels = axes[iax].get_legend_handles_labels()
        l = axes[iax].legend(handles[0:narea], labels[0:narea])
        axes[iax].set_ylabel(ylabel[what], labelpad=0.1)
        axes[iax].set_xlabel(xlabel[what], labelpad=0.1)

    yticks = [range(0, 5001, 1000), range(0, 501, 100)]
    for ix, x in enumerate(axes):
        x.set_ylim([0, x.get_ylim()[1]])
        x.tick_params(direction='in', width=1.5)
        x.tick_params(axis='y', pad=0.1)

        lab = [str(l) if not i % 2 else '' for i, l in enumerate(yticks[ix])]
        x.set_yticks(yticks[ix], labels=lab, rotation=90, va='center')
        x.set_ylim([0, yticks[ix][-1]])
        x.spines['top'].set_visible(False)
        x.spines['right'].set_visible(False)
        x.spines['left'].set_lw(1.5)
        x.spines['bottom'].set_lw(1.5)
    x.get_legend().remove()
    fig.subplots_adjust(right=0.98, top=0.98, left=0.15)
    fig.set_size_inches([2.5, 3])
    fig.savefig(processed / project / ('dilution_area_of_interest.png'), dpi=600)
    fig.savefig(processed / project / ('dilution_area_of_interest.svg'), dpi=600)
