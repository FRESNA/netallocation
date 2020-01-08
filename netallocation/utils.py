#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 15:10:07 2019

@author: fabian
"""

import pandas as pd
import xarray as xr
from pypsa.geo import haversine_pts
from sparse import as_coo

def upper(ds):
    ds = obj_if_acc(ds)
    return ds.clip(min=0)

def lower(ds):
    ds = obj_if_acc(ds)
    return ds.clip(max=0)

def get_branches_i(n, branch_components=None):
    if branch_components is None: branch_components = n.branch_components
    return pd.concat((n.df(c)[[]] for c in branch_components),
           keys=branch_components).index.rename(['component', 'branch_i'])

def filter_null(da, dim=None):
    da = obj_if_acc(da)
    if dim is not None:
        return da.where(da != 0).dropna(dim, how='all')
    return da.where(da != 0)

def array_as_sparse(da):
    return da.copy(data=as_coo(da.data))

def as_sparse(ds):
    ds = obj_if_acc(ds)
    if isinstance(ds, xr.Dataset):
        return ds.assign(**{k: array_as_sparse(ds[k]) for k in ds})
    else:
        return array_as_sparse(ds)

def array_as_dense(da):
    return da.copy(data=da.data.todense())

def as_dense(ds):
    ds = obj_if_acc(ds)
    if isinstance(ds, xr.Dataset):
        return ds.assign(**{k: array_as_dense(ds[k]) for k in ds})
    else:
        return array_as_dense(ds)


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
    xy = n.buses[['x', 'y']]
    d = xy.apply(lambda ds: pd.Series(haversine_pts(ds, xy), xy.index), axis=1)
    return xr.DataArray(d, dims=['source', 'sink'])


def convert_vip_to_p2p(ds):
    """
    Converts a virtual injection pattern into a peer-to-peer allocation.

    Parameters
    -----------
    ds : xarray.Dataset or xarray.DataArray

    Returns
    -------
    A xarray.Dataset with the peer-to-peer variable appended if a Dataset was
    passed, passes the converted DataArray if a DataArray was passed.

    """
    ds = obj_if_acc(ds)
    da = ds.virtual_injection_pattern if isinstance(ds, xr.Dataset) else ds
    p2p = upper(da.rename(injection_pattern='sink', bus='source') -
                da.rename(injection_pattern='source', bus='sink'))
    return ds.assign(peer_to_peer = p2p) if isinstance(ds, xr.Dataset) else p2p

