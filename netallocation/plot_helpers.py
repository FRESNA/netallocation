#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 09:52:39 2019

@author: fabian
"""

import numpy as np
import geopandas as gpd
from matplotlib.patches import Circle, Ellipse
from matplotlib.legend_handler import HandlerPatch
import pandas as pd
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
from pathlib import Path
import yaml

filename =  Path(__file__).parent.parent.joinpath('color_config.yaml')
with open(filename) as f:
    config = yaml.load(f, Loader=yaml.FullLoader)


plt.rcParams['font.size'] = 11.0

nice_names = {'offwind-ac': 'Offshore', 'offwind-dc': 'Offshore',
              'onwind': 'Onshore',
              'hydro': 'Hydro',
              'ror': 'Run-of-River', 'solar': 'Solar',
              'PHS': 'Pumped Hydro', 'H2': 'Hydrogen Storage',
              'battery': 'Battery',
              'OCGT': 'OCGT'}

light_gray = ".8"
style_dict = {"axes.facecolor": "white",
                "axes.edgecolor": light_gray,
                "grid.color": light_gray,

                "axes.spines.left": True,
                "axes.spines.bottom": True,
                "axes.spines.right": True,
                "axes.spines.top": True,
                "xtick.bottom": False,
                "ytick.left": False}

plt.rcParams['axes.edgecolor'] = plt.rcParams['grid.color']
plt.rcParams.update(style_dict)

fuel_colors = pd.Series(config['fuel_to_color'])

def make_legend_circles_for(sizes, scale=1.0, **kw):
    from matplotlib.patches import Circle
    return [Circle((0,0), radius=(s/scale)**0.5, **kw) for s in sizes]

def handles_labels_for(color_series, kind='circle', kw_list=None, **kw):
    color_series = pd.Series(color_series) if isinstance(color_series, dict) \
                     else color_series
    colors = color_series.values
    labels = color_series.index.tolist()
    if kw_list is None:
        kw_list = [{}]*len(colors)
    if kind == 'circle':
        handles = [Line2D([], [], c=c, marker='.', linestyle='None',
                          markersize=20, **k,  **kw)
                   for c, k in zip(colors, kw_list)]
#        handles = [Circle((0, 0), color=c, **k,  **kw)
#                    for c, k in zip(colors, kw_list)]
    elif kind == 'line':
        handles = [Line2D((0, 1), (0.5, 0.5), color=c, **k,  **kw)
                    for c, k in zip(colors, kw_list)]
    return handles, labels



def make_handler_map_to_scale_circles_as_in(ax, dont_resize_actively=False):
    fig = ax.get_figure()
    def axes2pt():
        return np.diff(ax.transData.transform([(0,0), (1,1)]), axis=0)[0] * \
                (72./fig.dpi)

    ellipses = []
    if not dont_resize_actively:
        def update_width_height(event):
            dist = axes2pt()
            for e, radius in ellipses: e.width, e.height = 2. * radius * dist
        fig.canvas.mpl_connect('resize_event', update_width_height)
        ax.callbacks.connect('xlim_changed', update_width_height)
        ax.callbacks.connect('ylim_changed', update_width_height)

    def legend_circle_handler(legend, orig_handle, xdescent, ydescent,
                              width, height, fontsize):
        w, h = 2. * orig_handle.get_radius() * axes2pt()
        e = Ellipse(xy=(0.5*width-0.5*xdescent, 0.5*height-0.5*ydescent),
                    width=w, height=w)
        ellipses.append((e, orig_handle.get_radius()))
        return e
    return {Circle: HandlerPatch(patch_func=legend_circle_handler)}


def load_regions(n, onshore=True):
    prepend = 'regions_onshore_' if onshore else 'regions_offshore_'
    name = n if isinstance(n, str) else n.name
    return gpd.read_file('models/' + prepend + '_'.join(name.split('_')[:3])
                         + '.geojson').set_index('name')


def adjust_breaking_axis(ax, ax2, break_y_axis_at=None):
    if break_y_axis_at is not None:
            ax.set_ylim(bottom=break_y_axis_at)
            ax2.set_ylim(top=break_y_axis_at)

    d = .02  # how big to make the diagonal lines in axes coordinates
    # arguments to pass to plot, just so we don't keep repeating them
    kwargs = dict(transform=ax.transAxes, color='black',
                  clip_on=False, linewidth=1.)
    ax.plot((-d, +d), (-d, +d), **kwargs, zorder=3)        # top-left diagonal
    ax.plot((1 - d, 1 + d), (-d, +d), **kwargs, zorder=3)  # top-right diagonal

    kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
    ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
    ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal
    ax.spines['bottom'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.set_title(None)


def lighten_color(color, amount=0.5):
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    copied from
    https://stackoverflow.com/questions/37765197/darken-or-lighten-a-color-in-matplotlib
    """
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])
