#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module comprises functions to convert different representations of
allocation into each other.
"""
import xarray as xr
import pandas as pd
from .utils import obj_if_acc, upper, check_dataset, is_sparse
from .grid import Incidence, self_consumption
from .linalg import dot

def virtual_patterns(ds, n, q=0.5):
    """
    Converts a peer-on-branch-to-peer into a virtual flow/injection pattern.

    If a virtual flow pattern array is already existent, nothing is done.

    Parameters
    -----------
    ds : xarray.Dataset or xarray.DataArray
    n : pypsa.Network
        Underlying network. This is needed for determining the Incidence matrix.
    q : float in [0,1]
        Ratio of net source and net sink contributions in the virtual patterns.
        If q=0 then only net importers are taken into account (if q=1 only
        net exporters).
    Returns
    -------
    A xarray.Dataset with the virtual flow pattern variable appended if a
    Dataset was passed, passes the converted DataArray if a DataArray was passed.

    """
    if 'virtual_flow_pattern' in ds and 'virtual_injection_pattern' in ds:
        return ds
    ds = obj_if_acc(ds)
    is_dataset = isinstance(ds, xr.Dataset)
    da = ds.peer_on_branch_to_peer if is_dataset else ds
    vfp = q * da.sum('sink').rename(source='bus') + \
          (1 - q) * da.sum('source').rename(sink='bus')
    K = Incidence(n, vfp.get_index('branch').unique('component'), is_sparse(ds))
    vip = K @ vfp.rename(bus='injection_pattern')
    virtuals = xr.Dataset({'virtual_flow_pattern': vfp.T,
                           'virtual_injection_pattern': vip.T})
    return ds.merge(virtuals) if is_dataset else virtuals


def vip_to_p2p(ds, n):
    """
    Converts a virtual injection pattern into a peer-to-peer allocation.

    If a peer-to-peer array is already existent, nothing is done.

    Parameters
    -----------
    ds : xarray.Dataset or xarray.DataArray

    Returns
    -------
    A xarray.Dataset with the peer-to-peer variable appended if a Dataset was
    passed, passes the converted DataArray if a DataArray was passed.

    """
    ds = obj_if_acc(ds)
    ds, was_ds = check_dataset(ds)
    if 'peer_to_peer' in ds:
        return ds if was_ds else ds['peer_to_peer']
    vip = ds.virtual_injection_pattern
    p2p = upper(vip.rename(injection_pattern='sink', bus='source') -
                vip.rename(injection_pattern='source', bus='sink'))
    s = self_consumption(n, ds.snapshot)
    s['bus'] = pd.MultiIndex.from_tuples([(b,b) for b in s['bus'].values],
                                         names=['source', 'sink'])
    p2p += s.unstack('bus', fill_value=0).reindex_like(p2p)
    return ds.assign(peer_to_peer = p2p) if was_ds else p2p
