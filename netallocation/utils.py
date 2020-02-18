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
from .decorators import check_branch_components

def upper(ds):
    "Clip all negative entries of a xr.Dataset/xr.DataArray."
    ds = obj_if_acc(ds)
    return ds.clip(min=0)

def lower(ds):
    "Clip all positive entries of a xr.Dataset/xr.DataArray."
    ds = obj_if_acc(ds)
    return ds.clip(max=0)

@check_branch_components
def get_branches_i(n, branch_components=None):
    "Get a pd.Multiindex for all branches in the Network."
    return pd.concat((n.df(c)[[]] for c in branch_components),
           keys=branch_components).index.rename(['component', 'branch_i'])

def filter_null(da, dim=None):
    "Drop all coordinates with only null/nan entries on dimensions dim."
    da = obj_if_acc(da)
    if dim is not None:
        return da.where(da != 0).dropna(dim, how='all')
    return da.where(da != 0)

def array_as_sparse(da):
    "Convert a dense xr.DataArray to a sparse dataarray."
    return da.copy(data=as_coo(da.data))

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
    if isinstance(ds, xr.Dataset):
        return ds.assign(**{k: array_as_sparse(ds[k]) for k in ds})
    else:
        return array_as_sparse(ds)

def array_as_dense(da):
    "Convert a sparse xr.DataArray to a dense dataarray."
    return da.copy(data=da.data.todense())

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
    if isinstance(ds, xr.Dataset):
        return ds.assign(**{k: array_as_dense(ds[k]) for k in ds})
    else:
        return array_as_dense(ds)

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


def group_per_bus_carrier(df, c, n):
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
    df : pd.DataFrame
        Grouped dataframe with Multiindex ('bus', 'carrier').

    """
    df = df.groupby(n.df(c)[['bus', 'carrier']].apply(tuple, axis=1), axis=1).sum()
    df.columns = pd.MultiIndex.from_tuples(df.columns, names=['bus', 'carrier'])
    return df
