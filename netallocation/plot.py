#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 16:13:57 2019

@author: fabian
"""

import pandas as pd
import matplotlib.pyplot as plt
import pypsa
import cartopy.crs as ccrs
import numpy as np
from . import plot_helpers


def replace_carrier_names(n, replace=None):
    if replace is None: replace = plot_helpers.nice_names
    n.loads = n.loads.assign(carrier='Load')
    n.generators.carrier.replace(replace, inplace=True)
    n.storage_units.carrier.replace(replace, inplace=True)
    n.carriers = n.carriers.rename(index=replace)\
                   [lambda ds: ~ds.index.duplicated()]



def chord_diagram(allocation, agg='mean', minimum_quantile=0,
                  groups=None, size=200, pallette='Category20',
                  fig_inches=4):
    """
    This function builds a chord diagram on the base of holoviews [1].
    It visualizes allocated peer-to-peer flows for all buses given in
    the data. As for compatibility with ipython shell the rendering of
    the image is passed to matplotlib however to the disfavour of
    interactivity. Note that the plot becomes only meaningful for networks
    with N > 5, because of sparse flows otherwise.


    [1] http://holoviews.org/reference/elements/bokeh/Chord.html

    Parameters
    ----------

    allocation : pandas.Series (MultiIndex)
        Series of power transmission between buses. The first index
        level ('source') represents the source of the flow, the second
        level ('sink') its sink.
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

    if len(allocation.index.levels) == 3:
        allocation = allocation.agg('mean', level=['source', 'sink'])

    allocated_buses = allocation.index.levels[0] \
                      .append(allocation.index.levels[1]).unique()
    bus_map = pd.Series(range(len(allocated_buses)), index=allocated_buses)

    links = allocation.to_frame('value').reset_index()\
        .replace({'source': bus_map, 'sink': bus_map})\
        .sort_values('source').reset_index(drop=True) \
        [lambda df: df.value >= df.value.quantile(minimum_quantile)]

    nodes = pd.DataFrame({'bus': bus_map.index})
    if groups is None:
        cindex = 'index'
        ecindex = 'source'
    else:
        groups = groups.rename(index=bus_map)
        nodes = nodes.assign(groups=groups)
        links = links.assign(groups=links['source']
                             .map(groups))
        cindex = 'groups'
        ecindex = 'groups'

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
                                 'edge_color_index': ecindex
                                 })

#    fig = hv.render(diagram, size=size, dpi=300)

    fig = LayoutPlot(Layout([diagram]), dpi=300, fig_size=size,
                     fig_inches=fig_inches,
                     tight=True, tight_padding=0,
                     fig_bounds=(-.15, -.15, 1.15, 1.15),
                     hspace=0, vspace=0, fontsize=15)\
             .initialize_plot()
    return fig, fig.axes




def network_plot(n, linewidth_factor=5e3, bus_size_factor=1e5,
                 boundaries=None):
    import matplotlib as mpl
    to_rgba = mpl.colors.colorConverter.to_rgba

    line_colors = {'cur': "purple",
                   'exp': to_rgba("red", 0.5)}


    bus_sizes = pd.concat((n.generators.groupby(['bus', 'carrier'])
                            .p_nom_opt.sum(),
                           n.storage_units.groupby(['bus', 'carrier'])
                            .p_nom_opt.sum()))
    line_widths_exp = pd.concat(dict(Line=n.lines.s_nom_opt,
                                     Link=n.links.p_nom_opt))
    line_widths_cur = pd.concat(dict(Line=n.lines.s_nom_min,
                                     Link=n.links.p_nom_min))


    line_colors_with_alpha = (line_widths_cur['Line'] / n.lines.s_nom > 1e-3)\
                            .map({True: line_colors['cur'], False:
                              to_rgba(line_colors['cur'], 0.)})
    link_colors_with_alpha = (line_widths_cur['Link'] / n.links.p_nom > 1e-3)\
                             .map({True: line_colors['cur'], False:
                             to_rgba(line_colors['cur'], 0.)})

    ## PLOT
    from pypsa.plot import projected_area_factor
    fig, ax = plt.subplots(subplot_kw={"projection":ccrs.EqualEarth()},
                                       figsize=(8,7))

    n.plot(line_widths=line_widths_exp/linewidth_factor,
           line_colors=n.lines.assign(c = 'exp').c.map(line_colors),
           link_colors=n.links.assign(c = 'exp').c.map(line_colors),
           bus_sizes=bus_sizes/bus_size_factor,
           bus_colors=plot_helpers.fuel_colors,
           geomape=True,
           boundaries=boundaries,
           ax=ax)
    n.plot(line_widths=line_widths_cur/linewidth_factor,
           line_colors=line_colors_with_alpha,
           link_colors=link_colors_with_alpha,
           bus_sizes=0.,
           basemap=True, basemap_kwargs={'border':False, 'coastline':False},
           legend=False,
           boundaries=boundaries,
           ax=ax)
    ax.axis('off')
    ax.artists[4].set_title('Carriers')

    # LEGEND add capcacities
    reference_caps = [10e3, 5e3, 1e3]
    handles = plot_helpers.make_legend_circles_for(reference_caps,
                            scale=bus_size_factor / boundary_area_factor(ax)**2,
                            facecolor="w", edgecolor='grey')
    labels = ["{} GW".format(int(s/1e3)) for s in reference_caps]
    l2 = ax.legend(handles, labels,
                   loc="upper left", bbox_to_anchor=(1, 0.5),
                   facecolor='w',
                   frameon=True, edgecolor='w',
                   title='Capacity',
                   handler_map=plot_helpers
                       .make_handler_map_to_scale_circles_as_in(ax))
    ax.add_artist(l2)

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

    ax.add_artist(ax.legend(handles, labels, edgecolor='w',
                  facecolor='inherit', fancybox=True,
                  loc="lower left", bbox_to_anchor=(1, .0),
                  frameon=True, ncol=2, columnspacing=0.5,
                  title='Transmission Exist./Exp.'))

    fig.canvas.draw(); fig.tight_layout()
    return fig, ax


def component_plot(n, linewidth_factor=5e3, gen_size_factor=5e4, figsize=(10, 5),
                   sus_size_factor=1e4, boundaries=[-10. , 30, 36, 70]):
    import matplotlib as mpl
    to_rgba = mpl.colors.colorConverter.to_rgba

    line_colors = {'cur': "purple",
                   'exp': to_rgba("red", 0.5)}

    gen_sizes = n.generators.groupby(['bus', 'carrier']).p_nom_opt.sum()
    store_sizes = n.storage_units.groupby(['bus', 'carrier']).p_nom_opt.sum()\
                   .append(n.stores.groupby(['bus', 'carrier']).e_nom_opt.sum())

    branch_widths = pd.concat([n.lines.s_nom_min, n.links.p_nom_min],
                              keys=['Line', 'Link']).div(linewidth_factor)

    ## PLOT
    from pypsa.plot import projected_area_factor
    fig, (ax, ax2)  = plt.subplots(1, 2, figsize=figsize,
          subplot_kw={"projection":ccrs.EqualEarth()})

    n.plot(bus_sizes=gen_sizes/gen_size_factor,
           bus_colors=plot_helpers.fuel_colors.to_dict(),
           line_widths=branch_widths,
           line_colors={'Line':line_colors['cur'], 'Link': line_colors['cur']},
           geomap=True,
           boundaries=boundaries,
           title = 'Generation \& Transmission Capacities',
           ax=ax)

    branch_widths = pd.concat([n.lines.s_nom_opt-n.lines.s_nom_min,
                               n.links.p_nom_opt-n.links.p_nom_min],
                              keys=['Line', 'Link']).div(linewidth_factor)

    n.plot(bus_sizes=store_sizes/sus_size_factor,
           bus_colors=plot_helpers.fuel_colors.to_dict(),
           line_widths=branch_widths,
           line_colors=n.branches().assign(c = 'exp').c.map(line_colors),
           geomap=True,
           boundaries=boundaries,
           title='Storages Capacities \& Transmission Expansion',
           ax=ax2)
    ax.axis('off')
#    ax.artists[2].set_title('Carriers')

    # LEGEND add capcacities
    for axis, scale in zip((ax, ax2), (gen_size_factor, sus_size_factor)):
        reference_caps = [10e3, 5e3, 1e3]
        handles = plot_helpers.make_legend_circles_for(reference_caps,
                                          scale=scale /
                                              projected_area_factor(axis)**2,
                                          facecolor="w", edgecolor='grey')
        labels = ["{} GW".format(int(s/1e3)) for s in reference_caps]
        l2 = axis.legend(handles, labels, framealpha=0.7,
                       loc="upper left", bbox_to_anchor=(0., 1),
                       frameon=True, #edgecolor='w',
                       title='Capacity',
                       handler_map=plot_helpers
                           .make_handler_map_to_scale_circles_as_in(axis))
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

    fig.add_artist(fig.legend(handles, labels,
                  loc="lower left", bbox_to_anchor=(1., .0),
                  frameon=False,
                  ncol=2, columnspacing=0.5,
                  title='Transmission Exist./Exp.'))

    # legend generation colors
    colors = plot_helpers.fuel_colors[n.generators.carrier.unique()]
    fig.add_artist(fig.legend(*plot_helpers.handles_labels_for(colors),
                              loc='upper left', bbox_to_anchor=(1, 1),
                              frameon=False,
                              title='Generation carrier'))
    # legend storage colors
    fig.add_artist(
            fig.legend(*plot_helpers.handles_labels_for(
                    plot_helpers.fuel_colors[n.storage_units.carrier.unique()]),
               loc='center left', bbox_to_anchor=(1, 0.5),
               frameon=False,
               title='Storage carrier'))

    fig.canvas.draw(); fig.tight_layout(pad=0.5)
    return fig, (ax, ax2)


def fact_sheet(n, fn_out):
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

    d['Curtailment per Carrier']  = (n.generators_t.p_max_pu
                                     * n.generators.p_nom_opt -
                                     n.generators_t.p).dropna(axis=1).sum()\
                                     .groupby(carriers).sum()\
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
    df =  df.to_frame('value')
    df.to_html(f'{fn_out}fact_sheet.html')
    return df
