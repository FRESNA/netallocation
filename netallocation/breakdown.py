#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 15:38:24 2019

@author: fabian
"""

from .grid import power_demand, power_production, network_injection
from .utils import (as_sparse, obj_if_acc, is_sparse, check_dataset,
                    check_carriers, check_snapshots)
from .convert import vip_to_p2p
from sparse import COO
import logging
import xarray as xr
from dask.diagnostics import ProgressBar

logger = logging.getLogger(__name__)


def expand_by_source_type(ds, n, chunksize=None, dim='source'):
    """
    Breakdown allocation into generation carrier type.

    These include carriers of all components specified by 'components'.
    Note that carrier names of all components have to be unique.

    Parameters
    ----------
    ds : xarray.Dataset
        Allocation Data with dimension 'source'
    n : pypsa.Network()
        Network which the allocation was derived from
    chunksize : int
        Chunksize of the snapshot chunks passed to dask for computing faster
        and with less memory usage for large datasets.
    dim : str
        Name of dimension to be expanded by carrier (must contain bus names).

    Example
    -------
    >>> ap = ntl.flow_allocation(n, n.snapshots, method='ap')
    >>> ntl.breakdown.expand_by_source_carrier(ap, n)

    """
    ds = obj_if_acc(ds)
    if 'source_carrier' in ds.dims:
        return ds
    ds, was_ds = check_dataset(ds)
    sns = check_snapshots(ds.snapshot, n)
    prod = power_production(n, sns, per_carrier=True)
    share = (prod / prod.sum('carrier'))\
             .rename(bus=dim, carrier='source_carrier').fillna(0)
    assert dim in ds.dims, f'Dimension {dim} not present in Dataset'
    expand = ds[[k for k in ds if dim in ds[k].dims]]
    if is_sparse(expand):
        share = as_sparse(share.fillna(0))
    if any(isinstance(expand[v], COO) for v in expand):
        TypeError('All variables of the dataset must either be sparse or dense.')

    logger.info('Expanding by source carrier')
    if chunksize is None:
        res = expand * share
    else:
        chunk = {'snapshot': chunksize}
        with ProgressBar():
            res = (expand.chunk(chunk) * share.chunk(chunk)).compute()
    if was_ds:
        return res.merge(ds, compat='override', join='left').assign_attrs(ds.attrs)
    return res[list(res)[0]]


def expand_by_sink_type(ds, n, chunksize=None, dim='source'):
    """
    Breakdown allocation into demand types, e.g. Storage carriers and Load.

    These include carriers of all components specified by 'components'. Note
    that carrier names of all components have to be unique.

    Parameters
    ----------
    ds : xarray.Dataset
        Allocation Data with dimension 'sink'
    n : pypsa.Network
        Network which the allocation was derived from
    chunksize : int
        Chunksize of the snapshot chunks passed to dask for computing faster
        and with less memory usage for large datasets.
    dim : str
        Name of dimension to be expanded by carrier (must contain bus names).

    Example
    -------
    >>> ap = ntl.flow_allocation(n, n.snapshots, method='ap')
    >>> ntl.breakdown.expand_by_sink_carrier(ap, n)

    """
    ds = obj_if_acc(ds)
    if 'sink_carrier' in ds.dims:
        return ds
    ds, was_ds = check_dataset(ds)
    sns = check_snapshots(ds.snapshot, n)
    demand = power_demand(n, sns, per_carrier=True)
    share = (demand / demand.sum('carrier'))\
             .rename(bus=dim, carrier='sink_carrier')
    assert dim in ds.dims, f'Dimension {dim} not present in Dataset'
    expand = ds[[k for k in ds if dim in ds[k].dims]]
    if is_sparse(expand):
        share = as_sparse(share.fillna(0))
    if any(isinstance(expand[v], COO) for v in expand):
        TypeError('All variables of the dataset must either be sparse or dense.')

    logger.info('Expanding by sink carrier')
    if chunksize is None:
        res = expand * share
    else:
        chunk = {'snapshot': chunksize}
        with ProgressBar():
            res = (expand.chunk(chunk) * share.chunk(chunk)).compute()
    if was_ds:
        return res.merge(ds, compat='override', join='left').assign_attrs(ds.attrs)
    return res[list(res)[0]]


def by_carriers(ds, n, chunksize=None):
    """
    Breakdown allocation into production and demand carriers.

    Use this funtion to breakdown the share of single production carriers
    (carriers of generators, storage units, stores) and demand carriers
    (carriers of loads, storage units, stores). Note that carrier names of all
    components have to be unique. The funtion will return the a dataset or
    dataarray with two additional dimensions, `source_carrier` and
    `sink_carrier`.


    Parameters
    ----------
    ds : xarray.Dataset
        Allocation Data with dimension `sink` and `source`
    n : pypsa.Network
        Network which the allocation was derived from
    chunksize : int
        Chunksize of the snapshot chunks passed to dask for computing faster
        and with less memory usage for large datasets.

    Example
    -------
    >>> ap = ntl.flow_allocation(n, n.snapshots, method='ap')
    >>> ntl.breakdown.carriers(ap, n)

    """
    ds = obj_if_acc(ds)
    if 'sink_carrier' not in ds.dims:
        ds = expand_by_sink_type(ds, n, chunksize)
    if 'source_carrier' not in ds.dims:
        ds = expand_by_source_type(ds, n, chunksize)
    return ds