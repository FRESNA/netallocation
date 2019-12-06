#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 15:10:07 2019

@author: fabian
"""

import pandas as pd
import logging
logger = logging.getLogger(__name__)
import pypsa

compute_if_dask = lambda df, b: df.compute() if b else df

def upper(df):
    return df.clip(lower=0)

def lower(df):
    return df.clip(upper=0)


def get_branches_i(n, branch_components=None):
    if branch_components is None: branch_components = n.branch_components
    return pd.concat((n.df(c)[[]] for c in branch_components),
           keys=branch_components).index.rename(['component', 'branch_i'])

def last_to_first_level(ds):
    return ds.reorder_levels([ds.index.nlevels-1] + \
                             list(range(ds.index.nlevels-1)))\
             .sort_index(level=0, sort_remaining=False)

def to_dask(df, use_dask=False):
    if use_dask:
        import dask.dataframe as dd
        if df.index.names[0] == 'snapshot':
            return dd.from_pandas(df, npartitions=1).repartition(freq='1m')
        else:
            npartitions = 1+df.memory_usage(deep=True).sum() // 100e6
            return dd.from_pandas(df, npartitions=npartitions)
    else:
        return df


def _to_categorical_index(df, axis=0):

    def to_cat_if_obj(i):
        return i.astype('category') if i.is_object() else i

    if df.axes[axis].nlevels > 1:
        return df.set_axis(
                df.axes[axis]
                    .set_levels([to_cat_if_obj(i) for i
                                 in df.axes[axis].levels]),
                inplace=False, axis=axis)
    else:
        if df.axes[axis].is_object():
            return df.set_axis(to_cat_if_obj(df.axes[axis]),
                inplace=False, axis=axis)


def _sync_categrorical_axis(df1, df2, axis=0):
    overlap_levels = [n for n in df1.axes[axis].names
                      if n in df2.axes[axis].names and
                      (df1.axes[axis].unique(n).is_categorical() &
                       df2.axes[axis].unique(n).is_categorical())]
    union = [df1.axes[axis].unique(n).union(df2.axes[axis].unique(n))
             .categories for n in overlap_levels]
    for level, cats in zip(overlap_levels, union):
        df1 = df1.pipe(_set_categories_for_level, level, cats, axis=axis)
        df2 = df2.pipe(_set_categories_for_level, level, cats, axis=axis)


def _set_categories_for_level(df, level, categories, axis=0):
    level = [level] if isinstance(level, str) else level
    return df.set_axis(
            df.axes[axis].set_levels([i.set_categories(categories)
            if i.name in level else i for i in df.axes[axis].levels]),
        inplace=False, axis=axis)


def set_cats(df, n=None, axis=0):
    """
    Helper function for converting index of allocation series to categoricals.
    If a network is passed the categories will be aligned to the components
    of the network.
    """
    if n is None:
        return df.pipe(_to_categorical_index, axis=axis)
    buses = n.buses.index
    branch_i = n.branches().index.levels[1]
    bus_lv_names = ['sink', 'source', 'bus0', 'bus1', 'in', 'out']
    return df.pipe(_to_categorical_index, axis=axis)\
             .pipe(_set_categories_for_level, bus_lv_names, buses, axis=axis)\
             .pipe(_set_categories_for_level, ['branch_i'],
                   branch_i, axis=axis)

def droplevel(df, levels, axis=0):
    ax = df.axes[axis]
    for level in levels:
        ax = ax.droplevel(level)
    return df.set_axis(ax, axis=axis, inplace=False)


def parmap(f, arg_list, nprocs=None, **kwargs):
    import multiprocessing

    def fun(f, q_in, q_out):
        while True:
            i, x = q_in.get()
            if i is None:
                break
            q_out.put((i, f(x, **kwargs)))

    if nprocs is None:
        nprocs = multiprocessing.cpu_count()
    logger.info('Run process with {} parallel threads.'.format(nprocs))
    q_in = multiprocessing.Queue(1)
    q_out = multiprocessing.Queue()

    proc = [multiprocessing.Process(target=fun, args=(f, q_in, q_out))
            for _ in range(nprocs)]
    for p in proc:
        p.daemon = True
        p.start()

    sent = [q_in.put((i, x)) for i, x in enumerate(arg_list)]
    [q_in.put((None, None)) for _ in range(nprocs)]
    res = [q_out.get() for _ in range(len(sent))]
    [p.join() for p in proc]
    return [x for i, x in sorted(res)]


def get_test_network(linear=True):
    if linear:
        n = pypsa.Network(__file__ +'/../../testnetwork.nc')
        n.calculate_dependent_values(); n.determine_network_topology()
    else:
        # get solved scigrid model from pypsa example
        n = pypsa.Network(__file__ +'/../../testnetwork2.nc')
#        n.generators_t.p_set = n.generators_t.p
#        n.generators.control  = 'PV'
#        pq_gens = n.generators.query('bus == "AL0 0"').index
#        n.generators.loc[pq_gens, 'control'] = 'PQ'
#        n.generators_t.q_set.reindex_like(n.generators_t.p)\
#                        .assign(**{g: 0 for g in pq_gens})
#        n.pf(n.snapshots[0])
    return n
