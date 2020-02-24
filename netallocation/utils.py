#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module contains tool functions for making the life easier in the other
modules.
"""

import pandas as pd
import xarray as xr
from pypsa.geo import haversine_pts
from sparse import as_coo, COO

def upper(ds):
    "Clip all negative entries of a xr.Dataset/xr.DataArray."
    ds = obj_if_acc(ds)
    return ds.clip(min=0)

def lower(ds):
    "Clip all positive entries of a xr.Dataset/xr.DataArray."
    ds = obj_if_acc(ds)
    return ds.clip(max=0)

def get_branches_i(n, branch_components=None):
    "Get a pd.Multiindex for all branches in the Network."
    branch_components = check_branch_comps(branch_components, n)
    return pd.concat((n.df(c)[[]] for c in branch_components),
           keys=branch_components).index.rename(['component', 'branch_i'])

def filter_null(da, dim=None):
    "Drop all coordinates with only null/nan entries on dimensions dim."
    da = obj_if_acc(da)
    if dim is not None:
        return da.where(da != 0).dropna(dim, how='all')
    return da.where(da != 0)

def as_sparse(ds):
    """
    Convert dense dataset/dataarray into a sparse dataset.

    Parameters
    ----------
    ds : xr.Dataset or xr.DataArray

    Returns
    -------
    xr.Dataset or xr.DataArray
        Dataset or DataArray with sparse data.

    """
    ds = obj_if_acc(ds)
    func = lambda data: COO(data) if not isinstance(data, COO) else data
    return xr.apply_ufunc(func, ds)


def as_dense(ds):
    """
    Convert sparse dataset/dataarray into a dense dataset.

    Parameters
    ----------
    ds : xr.Dataset or xr.DataArray

    Returns
    -------
    xr.Dataset or xr.DataArray
        Dataset or DataArray with dense data.

    """
    ds = obj_if_acc(ds)
    func = lambda data: data.todense() if isinstance(data, COO) else data
    return xr.apply_ufunc(func, ds)

def is_sparse(ds):
    """
    Check if a xarray.Dataset or a xarray.DataArray is sparse.

    Parameters
    ----------
    ds : xarray.Dataset or xarray.DataArray

    Returns
    -------
    Bool

    """
    if isinstance(ds, xr.Dataset):
        return all(isinstance(ds[v].data, COO) for v in ds)
    else:
        return isinstance(ds.data, COO)


def obj_if_acc(obj):
    """
    Get the object of underying Accessor if Accessor is passed.

    This function is usefull to straightforwardly make functions through the
    AllocationAccessor accessible.

    Parameters
    ----------
    obj : AllocationAccessor or xarray.Dataset

    Returns
    -------
    obj
        Dataset of the accessor if accessor was is passed, ingoing object
        otherwise.

    """
    from .common import AllocationAccessor
    if isinstance(obj, AllocationAccessor):
        return obj._obj
    else:
        return obj

def bus_distances(n):
    """
    Calculate the geographical distances between all buses.

    Parameters
    ----------
    n : pypsa.Network

    Returns
    -------
    xr.DataArray
        Distance matrix of size N x N, with N being the number of buses.

    """
    xy = n.buses[['x', 'y']]
    d = xy.apply(lambda ds: pd.Series(haversine_pts(ds, xy), xy.index), axis=1)
    return xr.DataArray(d, dims=['source', 'sink'])


def reindex_by_bus_carrier(df, c, n):
    """
    Group a time-dependent dataframe by bus and carrier.

    Parameters
    ----------
    df : pd.DataFrame
        Time-dependent series to group, e.g. n.generators_t.p
    c : str
        Component name of the underlying data.
    n : pypsa.Network

    Returns
    -------
    xarray.DataArray
        Grouped array with dimension ('snapshot', 'bus', 'carrier').

    """
    check_duplicated_carrier(n)
    df = df.rename(columns=n.df(c)[['bus', 'carrier']].apply(tuple, axis=1))
    df.columns = pd.MultiIndex.from_tuples(df.columns, names=['bus', 'carrier'])
    return xr.DataArray(df).unstack('dim_1', fill_value=0)


def check_duplicated_carrier(n):
    check_carriers(n)
    dupl = pd.Series({c: n.df(c).carrier.unique() for c in n.one_port_components
                      if 'carrier' in n.df(c)}).explode()\
                      [lambda ds: ds.duplicated(keep=False)].dropna()
    assert dupl.empty, (f'The carrier name(s) {dupl.to_list()} appear in more '
                        f'than one component {dupl.index}. This will not work '
                        'when spanning the bus x carrier dimensions. Please '
                        'ensure unique carrier names.')


def check_carriers(n):
    """
    Ensure if carrier of stores is defined.

    Parameters
    ----------
    n : pypsa.Network

    """
    if 'carrier' not in n.loads:
        n.loads['carrier'] = 'Load'
    if 'carrier' not in n.stores:
        n.stores['carrier'] = n.stores.bus.map(n.buses.carrier)

def check_snapshots(arg, n):
    """
    Set argument to n.snapshots if None
    """
    if isinstance(arg, pd.Index):
        return arg.rename('snapshot')
    return n.snapshots.rename('snapshot') if arg is None else arg

def set_default_if_none(arg, n, attr):
    """
    Set any argument to an attribute of n if None
    """
    return getattr(n, attr) if arg is None else arg

def check_passive_branch_comps(arg, n):
    """
    Set argument to n.passive_branch_components if None
    """
    return set_default_if_none(arg, n, 'passive_branch_components')

def check_branch_comps(arg, n):
    """
    Set argument to n.branch_components if None
    """
    return set_default_if_none(arg, n, 'branch_components')

def check_one_port_comps(arg, n):
    """
    Set argument to n.one_port_components if None
    """
    return set_default_if_none(arg, n, 'one_port_components')

