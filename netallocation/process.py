#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 20:51:24 2019

@author: fabian
"""

import xarray as xr
import pandas as pd
from sparse import COO
from .utils import as_dense


def ensure_time_average(ds):
    if 'snapshot' in ds.dims:
        return ds.mean('snapshot')
    else:
        ds

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


def carrier_to_carrier(ds, split_local_nonlocal=False):
    ds = ensure_time_average(ds)
    if not split_local_nonlocal:
        return as_dense(ds.peer_to_peer.sum(['source', 'sink']))
    local = as_dense(local_consumption(ds).sum(['source', 'sink']))
    non_local = as_dense(nonlocal_consumption(ds).sum(['source', 'sink']))
    index = pd.Index(['local', 'nonlocal'], name='consumption_type')
    return xr.concat([local, non_local], dim=index)


def carrier_to_branch(ds):
    ds = ensure_time_average(ds)
    return ds.peer_on_branch_to_peer.sum(['source', 'sink'])


