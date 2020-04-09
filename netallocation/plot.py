#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 16:13:57 2019

@author: fabian
"""

from .plot_helpers import make_legend_circles_for, fuel_colors, \
    make_handler_map_to_scale_circles_as_in, handles_labels_for
from .utils import as_dense, filter_null
from pypsa.plot import projected_area_factor
import pandas as pd
import matplotlib.pyplot as plt
import pypsa
import numpy as np
from matplotlib.colors import to_hex, to_rgba
import logging
logger = logging.getLogger(__file__)


def chord_diagram(ds, agg='mean', minimum_quantile=0,
                  groups=None, size=200, pallette='Category20',
                  fig_inches=4):
    """
    Build a chord diagram on the base of holoviews [1].

    It visualizes allocated peer-to-peer flows for all buses given in
    the data. As for compatibility with ipython shell the rendering of
    the image is passed to matplotlib however to the disfavour of
    interactivity. Note that the plot becomes only meaningful for networks
    with N > 5, because of sparse flows otherwise.


    [1] http://holoviews.org/reference/elements/bokeh/Chord.html

    Parameters
    ----------
    allocation : xarray.Dataset
        Dataset with 'peer_to_peer' variable.
    lower_bound : int, default is 0
        filter small power flows by a lower bound
    groups : pd.Series, default is None
        Specify the groups of your buses, which are then used for coloring.
        The series must contain values for all allocated buses.
    size : int, default is 300
        Set the size of the holoview figure
    save_path : str, default is '/tmp/chord_diagram_pypsa'
        set the saving path of your figure

    """
    from holoviews.plotting.mpl import Layout, LayoutPlot
    import holoviews as hv
    hv.extension('matplotlib')

    allocation = filter_null(as_dense(ds.peer_to_peer.mean('snapshot')), 'source')\
                .to_series().dropna()
    if not groups is None:
        allocation = allocation.rename(groups).sum(level=['sink', 'source'])
    allocated_buses = allocation.index.levels[0] \
                      .append(allocation.index.levels[1]).unique()
    bus_map = pd.Series(range(len(allocated_buses)), index=allocated_buses)

    links = allocation.to_frame('value').reset_index()\
        .replace({'source': bus_map, 'sink': bus_map})\
        .sort_values('source').reset_index(drop=True) \
        [lambda df: df.value >= df.value.quantile(minimum_quantile)]

    nodes = pd.DataFrame({'bus': bus_map.index})
    cindex = 'index'
    ecindex = 'source'

    #annoying work around to construct cycler
    cmap = hv.plotting.util.process_cmap(pallette, ncolors=20)
    cmap = hv.core.options.Cycle(cmap)
    nodes = hv.Dataset(nodes, 'index')
    diagram = hv.Chord((links, nodes))
    diagram = diagram.opts(style={'cmap': cmap,
                                  'edge_cmap': cmap,
                                  'tight':True},
                           plot={'label_index': 'bus',
                                 'color_index': cindex,
                                 'edge_color_index': ecindex})
#    fig = hv.render(diagram, size=size, dpi=300)
    fig = LayoutPlot(Layout([diagram]), dpi=300, fig_size=size,
                     fig_inches=fig_inches,
                     tight=True, tight_padding=0,
                     fig_bounds=(-.15, -.15, 1.15, 1.15),
                     hspace=0, vspace=0, fontsize=15)\
             .initialize_plot()
    return fig, fig.axes

european_bounds = [-10. , 30, 36, 70]

def component_plot(n, linewidth_factor=5e3, gen_size_factor=5e4,
                   sus_size_factor=1e4,
                   carrier_colors=None, carrier_names=None,
                   figsize=(10, 5), boundaries=None,
                   **kwargs):
    """
    Plot a pypsa.Network generation and storage capacity

    Parameters
    ----------
    n : pypsa.Network
        Optimized network
    linewidth_factor : float, optional
        Scale factor of line widths. The default is 5e3.
    gen_size_factor : float, optional
        Scale factor of generator capacities. The default is 5e4.
    sus_size_factor : float, optional
        Scale factor of storage capacities. The default is 1e4.
    carrier_colors : pd.Series, optional
        Colors of the carriers. The default is None.
    carrier_names : pd.Series, optional
        Nice names of the carriers. The default is None.
    figsize : tuple, optional
        figsize of resulting image. The default is (10, 5).
    boundaries : tuple, optional
        Geographical bounds of the geomap. The default is [-10. , 30, 36, 70].

    Returns
    -------
    fig, ax
        Figure and axes of the corresponding plot.

    """
    if carrier_colors is None:
        carrier_colors = n.carriers.color
        fallback = pd.Series(n.carriers.index.str.title(), n.carriers.index)
        carrier_names = n.carriers.nice_name.replace('', np.nan).fillna(fallback)

    line_colors = {'cur': "purple", 'exp': to_hex(to_rgba("red", 0.5), True)}
    gen_sizes = n.generators.groupby(['bus', 'carrier']).p_nom_opt.sum()
    store_sizes = n.storage_units.groupby(['bus', 'carrier']).p_nom_opt.sum()
    branch_widths = pd.concat([n.lines.s_nom_min, n.links.p_nom_min],
                              keys=['Line', 'Link']).div(linewidth_factor)

    ## PLOT
    try:
        import cartopy.crs as ccrs
        projection = ccrs.EqualEarth()
        kwargs.setdefault('geomap', '50m')
    except ImportError:
        projection = None
        logger.warn('Could not import cartopy, drawing map disabled')

    fig, (ax, ax2)  = plt.subplots(1, 2, figsize=figsize,
                                   subplot_kw={"projection":projection})
    n.plot(bus_sizes = gen_sizes/gen_size_factor,
           bus_colors = carrier_colors,
           line_widths = branch_widths,
           line_colors = {'Line':line_colors['cur'], 'Link': line_colors['cur']},
           boundaries = boundaries,
           title = 'Generation \& Transmission Capacities',
           ax=ax, **kwargs)

    branch_widths = pd.concat([n.lines.s_nom_opt-n.lines.s_nom_min,
                               n.links.p_nom_opt-n.links.p_nom_min],
                              keys=['Line', 'Link']).div(linewidth_factor)

    n.plot(bus_sizes = store_sizes/sus_size_factor,
           bus_colors = carrier_colors,
           line_widths = branch_widths,
           line_colors = {'Line':line_colors['exp'], 'Link': line_colors['exp']},
           boundaries = boundaries,
           title = 'Storages Capacities \& Transmission Expansion',
           ax = ax2, **kwargs)
    ax.axis('off')
#    ax.artists[2].set_title('Carriers')

    # LEGEND add capcacities
    for axis, scale in zip((ax, ax2), (gen_size_factor, sus_size_factor)):
        reference_caps = [10e3, 5e3, 1e3]
        handles = make_legend_circles_for(reference_caps, scale=scale /
                                          projected_area_factor(axis)**2,
                                          facecolor="w", edgecolor='grey',
                                          alpha=.5)
        labels = ["{} GW".format(int(s/1e3)) for s in reference_caps]
        l2 = axis.legend(handles, labels, framealpha=0.7,
                       loc="upper left", bbox_to_anchor=(0., 1),
                       frameon=True, #edgecolor='w',
                       title='Capacity',
                       handler_map = make_handler_map_to_scale_circles_as_in(axis))
        axis.add_artist(l2)


    # LEGEND Transmission
    handles, labels = [], []
    for s in (10, 5):
        handles.append(plt.Line2D([0],[0],color=line_colors['cur'],
                                  linewidth=s*1e3/linewidth_factor))
        labels.append("/")
    for s in (10, 5):
        handles.append(plt.Line2D([0],[0],color=line_colors['exp'],
                                  linewidth=s*1e3/linewidth_factor))
        labels.append("{} GW".format(s))

    fig.artists.append(fig.legend(handles, labels,
                                  loc="lower left", bbox_to_anchor=(1., .0),
                                  frameon=False,
                                  ncol=2, columnspacing=0.5,
                                  title='Transmission Exist./Exp.'))

    # legend generation colors
    colors = carrier_colors[n.generators.carrier.unique()]
    if not carrier_names is None:
        colors = colors.rename(carrier_names)
    fig.artists.append(fig.legend(*handles_labels_for(colors),
                                  loc='upper left', bbox_to_anchor=(1, 1),
                                  frameon=False,
                                  title='Generation carrier'))
    # legend storage colors
    colors = carrier_colors[n.storage_units.carrier.unique()]
    if not carrier_names is None:
        colors = colors.rename(carrier_names)
    fig.artists.append(fig.legend(*handles_labels_for(colors),
                                  loc='upper left', bbox_to_anchor=(1, 0.45),
                                  frameon=False,
                                  title='Storage carrier'))

    fig.canvas.draw(); fig.tight_layout(pad=0.5)
    return fig, (ax, ax2)


def annotate_bus_names(n, ax, shift=-0.012, size=12, color='darkslategray',
                       **kwargs):
    """
    Annotate names of buses plot.


    Parameters
    ----------
    n : pypsa.Network
    ax : matplotlib axis
    shift : float/tuple
        Shift of the text with respect to the x and y bus coordinate.
        The default is -0.012.
    size : float, optional
        Text size. The default is 12.
    color : string, optional
        Text color. The default is 'k'.
    **kwargs : dict
        Keyword arguments going to ax.text() function. For example:

        - transform=ccrs.PlateCarree()
        - bbox=dict(facecolor='white', alpha=0.5, edgecolor='None')

    """
    kwargs.setdefault('zorder', 8)
    for index in n.buses.index:
        x, y = n.buses.loc[index, ['x', 'y']] + shift
        text = ax.text(x, y, index, size=size, color=color, ha="center", va="center",
                **kwargs)
    return text


def annotate_branch_names(n, ax, shift=-0.012, size=12, color='k', prefix=True,
                          **kwargs):
    def replace_branche_names(s):
        return s.replace('Line', 'AC ').replace('Link', 'DC ')\
                .replace('component', 'Line Type').replace('branch_i', '')\
                .replace('branch\_i', '')

    kwargs.setdefault('zorder', 8)
    branches = n.branches()
    branches = branches.assign(**{'loc0x': branches.bus0.map(n.buses.x),
                             'loc0y': branches.bus0.map(n.buses.y),
                             'loc1x': branches.bus1.map(n.buses.x),
                             'loc1y' : branches.bus1.map(n.buses.y)})
    for index in branches.index:
        loc0x, loc1x, loc0y, loc1y = \
            branches.loc[index, ['loc0x', 'loc1x', 'loc0y', 'loc1y']]
        if prefix:
            index = replace_branche_names(' '.join(index))
        else:
            index = index[1]
        ax.text((loc0x+loc1x)/2 + shift, (loc0y+loc1y)/2 + shift, index,
                    size=size, color=color, **kwargs)


def fact_sheet(n, fn_out=None):
    """
    Create a fact sheet which summarizes the network.

    Parameters
    ----------
    n : pypsa.Network
        Optimized network

    Returns
    -------
    df : pandas.DataFrame
        Summary of the network.

    """
    efactor = 1e6 # give power in TWh
    hfactor = n.snapshot_weightings[0] # total energy in elapsed hours

    carriers = n.generators.carrier
    d = pypsa.descriptors.Dict()

    d['Unit'] = f'10^{int(np.log10(efactor * 1e6))} Wh'
    d['Capacity [unit * 10^3]'] = n.generators.groupby('carrier').p_nom_opt.sum()\
        .append(n.storage_units.groupby('carrier').p_nom_opt.sum())\
        / efactor *1e3
    d['Total Production'] = n.generators_t.p.sum().sum() / efactor * hfactor
    d['Production per Carrier'] = n.generators_t.p.sum()\
                                  .groupby(n.generators.carrier).sum()\
                                  / efactor * hfactor
    d['Rel. Production per Carrier'] = d['Production per Carrier']/\
                                 d['Production per Carrier'].sum()
    d['Curtailment per Carrier']  = (
        n.generators_t.p_max_pu * n.generators.p_nom_opt -
        n.generators_t.p).dropna(axis=1).sum().groupby(carriers).sum()\
        / efactor * hfactor
    d['Rel. Curtailment per Carrier'] = (d['Curtailment per Carrier']/\
                                        d['Production per Carrier']).dropna()
    d['Total Curtailment']  = d['Curtailment per Carrier'].sum()
    d['Rel. Curtailement'] = d['Total Curtailment'] / d['Total Production']
    #storages
    d['Effective Sus Inflow'] = (n.storage_units_t.inflow.sum().sum()
                                - n.storage_units_t.spill.sum().sum())\
                                / efactor * hfactor
    d['Sus Total Charging'] = - n.storage_units_t.p.clip(upper=0).sum().sum() \
                                / efactor * hfactor
    d['Sus Total Discharging'] = n.storage_units_t.p.clip(lower=0).sum().sum() \
                                / efactor * hfactor
    for k,i in d.items():
        i = i.to_dict() if isinstance(i, pd.Series) else i
        d[k] = i
    df = pd.Series(d).apply(pd.Series).rename(columns={0: ''}).stack()
    return df
