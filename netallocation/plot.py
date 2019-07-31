#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 16:13:57 2019

@author: fabian
"""

import pandas as pd
import matplotlib.pyplot as plt

def chord_diagram(allocation, agg='mean', minimum_quantile=0,
                  groups=None, size=200, pallette='Category20'):
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

    fig = LayoutPlot(Layout([diagram]), dpi=300, fig_size=100, fig_inches=6,
                     tight=True, tight_padding=0, fig_bounds=(0, 0, 1, 1),
                     hspace=0, vspace=0)\
             .initialize_plot()
    return fig, fig.axes
