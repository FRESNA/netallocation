#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module comprises functions to convert different representations of
allocation into each other.
"""
import xarray as xr
import pandas as pd
import warnings
from .utils import obj_if_acc, upper, check_dataset, is_sparse
from .grid import Incidence, self_consumption, power_demand, power_production
from .linalg import dot
import warnings


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
    K = Incidence(
        n,
        vfp.get_index('branch').unique('component'),
        is_sparse(ds))
    vip = K @ vfp.rename(bus='injection_pattern')
    attrs = {'q': q}
    virtuals = xr.Dataset({'virtual_flow_pattern': vfp.T.assign_attrs(
        attrs), 'virtual_injection_pattern': vip.T.assign_attrs(attrs)})
    return ds.merge(virtuals) if is_dataset else virtuals


def peer_to_peer(ds, n, aggregated=None, q=None):
    """
    Converts a virtual injection pattern into a peer-to-peer allocation.

    If a peer-to-peer array is already existent, nothing is done.

    Parameters
    -----------
    ds : xarray.Dataset or xarray.DataArray
    aggregated: boolean, defaut None
        Within the aggregated coupling scheme (obtained if set to True),
        power production and demand are 'aggregated' within the corresponding
        bus. Therefore only the net surplus or net deficit of a bus is
        allocated to other buses.
        Within the direct coupling scheme (if set to False), production and
        demand are considered independent of the bus, therefore the power
        production and demand are allocated to all buses at the same time.
        Defaults to ds.virtual_injection_pattern.attrs['aggregated'].
    q : float, default None
        Only necessary when aggregated if False. Sets the shift parameter q,
        which determines the share of contribution of production and demand.
        Defaults to ds.virtual_injection_pattern.attrs['q'].
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
    if aggregated is None:
        assert 'aggregated' in vip.attrs, ('No attribute "aggregated" in '
                                           '"virtual_injection_pattern". Please set it manually, or assign'
                                           ' it the DataArray.')
        aggregated = vip.attrs['aggregated']
    if not aggregated:
        assert 'q' in vip.attrs, ('No shift parameter "q" in attributes'
                                  ' of "virtual_injection_pattern".')
        q = vip.attrs['q']
        p2p = vip.rename(injection_pattern='sink', bus='source')
        s = (1 - q) * power_demand(n, ds.snapshot) - \
            q * power_production(n, ds.snapshot)
    else:
        p2p = upper(vip.rename(injection_pattern='sink', bus='source') -
                    vip.rename(injection_pattern='source', bus='sink'))
        s = self_consumption(n, ds.snapshot)

    s['bus'] = pd.MultiIndex.from_tuples([(b, b) for b in s['bus'].values],
                                         names=['source', 'sink'])
    p2p = p2p + s.unstack('bus', fill_value=0).reindex_like(p2p)
    return ds.assign(peer_to_peer=p2p) if was_ds else p2p


def vip_to_p2p(ds, n, direct=False):
    warnings.warn("The function 'vip_to_p2p' is deprecated, use "
                  "'peer_to_peer' instead.", DeprecationWarning, 2)
    return peer_to_peer(ds, n, direct)
