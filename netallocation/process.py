#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 20:51:24 2019

@author: fabian
"""

import xarray as xr
import pandas as pd
from sparse import COO
from .utils import as_dense, upper, lower, is_sparse, as_sparse
from .grid import network_flow
from .breakdown import by_carriers
from dask.diagnostics import ProgressBar
import logging

def ensure_time_average(ds):
    if 'snapshot' in ds.dims:
        return ds.mean('snapshot')
    else:
        return ds

def split_local_nonlocal_consumption(ds):
    index = pd.Index(['local', 'nonlocal'], name='consumption_type')
    return xr.concat([local_consumption(ds), nonlocal_consumption(ds)], dim=index)


def local_consumption(ds):
    ptp = ds.peer_to_peer
    if isinstance(ptp.data, COO):
        get_index_i = lambda k: ptp.dims.index(k)
        coords = ptp.data.coords
        b = coords[get_index_i('source')] == coords[get_index_i('sink')]
        new_data = COO(coords[:, b], ptp.data.data[b], ptp.shape)
        return ptp.copy(data=new_data)
    buses = ds.get_index('source').rename('bus')
    return xr.concat((ds.peer_to_peer.sel(source=b, sink=b) for b in buses),
                     dim='source')

def nonlocal_consumption(ds):
    ptp = ds.peer_to_peer
    if isinstance(ptp.data, COO):
        get_index_i = lambda k: ptp.dims.index(k)
        coords = ptp.data.coords
        b = coords[get_index_i('source')] != coords[get_index_i('sink')]
        new_data = COO(coords[:, b], ptp.data.data[b], ptp.shape)
        return ptp.copy(data=new_data)
    buses = ds.get_index('source').rename('bus')
    return xr.concat((ds.peer_to_peer.sel(source=b).drop_sel(sink=b)
                      for b in buses), dim='source')


def carrier_to_carrier(ds, n, split_local_nonlocal=False):
    ds = ensure_time_average(ds)
    ds = by_carriers(ds, n)
    if not split_local_nonlocal:
        return as_dense(ds.peer_to_peer.sum(['source', 'sink']))
    local = as_dense(local_consumption(ds).sum(['source', 'sink']))
    non_local = as_dense(nonlocal_consumption(ds).sum(['source', 'sink']))
    index = pd.Index(['local', 'nonlocal'], name='consumption_type')
    return xr.concat([local, non_local], dim=index)


def carrier_to_branch(ds, n):
    ds = ensure_time_average(ds)
    ds = by_carriers(ds, n)
    return ds.peer_on_branch_to_peer.sum(['source', 'sink'])


def consider_branch_extension_on_flow(ds, n, chunksize=30):
    orig_branch_cap = xr.DataArray(dims='branch',
            data = pd.concat([n.lines.s_nom_min, n.links.p_nom_min],
                      keys=['Line', 'Link'], names=('component', 'branch_i')))
    flow = network_flow(n, snapshots=ds.get_index('snapshot'))
    vfp = ds.virtual_flow_pattern
    # first extention flow
    flow_shares = vfp / flow.sel(snapshot=vfp.snapshot)
    extension_flow_pos = upper(flow - orig_branch_cap)
    extension_flow_neg = lower(flow + orig_branch_cap)
    extension_flow = extension_flow_pos + extension_flow_neg
    within_cap_flow = flow.where(abs(flow) < orig_branch_cap)

    if is_sparse(vfp):
        extension_flow = as_sparse(extension_flow)
        within_cap_flow = as_sparse(within_cap_flow)
    chunk = {'snapshot': chunksize}
    with ProgressBar(minimum=2.):
        extension_flow = (extension_flow.chunk(chunk)
                          * abs(flow_shares).chunk(chunk)).compute()
        within_cap_flow = (within_cap_flow.chunk(chunk)
                           * abs(flow_shares).chunk(chunk)).compute()

    return xr.Dataset({'on_extension': extension_flow,
                       'on_original_cap': within_cap_flow},
                      attrs = {'type': 'Extension flow allocation'
                               f'with method: {ds.attrs["method"]}'})



