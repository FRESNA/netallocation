#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 15:38:24 2019

@author: fabian
"""

from .grid import power_demand, power_production
from .utils import as_sparse, obj_if_acc
from sparse import COO
import logging
from dask.diagnostics import ProgressBar

logger = logging.getLogger(__name__)

def expand_by_source_type(ds, n, components=['Generator', 'StorageUnit'],
                          chunksize=None):
    """
    Breakdown allocation into generation carrier type.

    These include carriers of all components specified by 'components'.
    Note that carrier names of all components have to be unique.

    Parameter
    ----------
    ds : xarray.Dataset
        Allocation Data with dimension 'source'
    n : pypsa.Network()
        Network which the allocation was derived from
    components : list, default ['Generator', 'StorageUnit']
        List of considered components. Carrier types of these components are
        taken for breakdown.
    chunksize : int
        Chunksize of the snapshot chunks passed to dask for computing faster
        and with less memory usage for large datasets.

    Example
    -------
    >>> ap = ntl.flow_allocation(n, n.snapshots, method='ap')
    >>> ntl.breakdown.expand_by_source_carrier(ap, n)

    """
    ds = obj_if_acc(ds)
    sns = ds.get_index('snapshot')
    share = (power_production(n, sns, per_carrier=True) / power_production(n, sns))
    share = share.rename(bus='source', carrier='source_carrier')
    if all(isinstance(ds[v].data, COO) for v in ds):
        share = as_sparse(share.fillna(0))
    elif any(isinstance(ds[v], COO) for v in ds):
        TypeError('All variables of the dataset must either be sparse or dense.')

    logger.info('Expanding by source carrier')
    if chunksize is None:
        res = ds * share
    else:
        chunk = {'snapshot': chunksize}
        with ProgressBar():
            res = (ds.chunk(chunk) * share.chunk(chunk)).compute()
    return res.assign_attrs(ds.attrs)
            #.stack({'production': ('source', 'source_carrier')})


def expand_by_sink_type(ds, n, components=['Load', 'StorageUnit'],
                        chunksize=None):
    """
    Breakdown allocation into demand types, e.g. Storage carriers and Load.

    These include carriers of all components specified by 'components'. Note
    that carrier names of all components have to be unique.

    Parameter
    ----------
    ds : xarray.Dataset
        Allocation Data with dimension 'sink'
    n : pypsa.Network()
        Network which the allocation was derived from
    components : list, default ['Load', 'StorageUnit']
        List of considered components. Carrier types of these components are
        taken for breakdown.
    chunksize : int
        Chunksize of the snapshot chunks passed to dask for computing faster
        and with less memory usage for large datasets.

    Example
    -------
    >>> ap = ntl.flow_allocation(n, n.snapshots, method='ap')
    >>> ntl.breakdown.expand_by_sink_carrier(ap, n)

    """
    ds = obj_if_acc(ds)
    sns = ds.get_index('snapshot')
    share = (power_demand(n, sns, per_carrier=True) / power_demand(n, sns))
    share = share.rename(bus='sink', carrier='sink_carrier')
    if all(isinstance(ds[v].data, COO) for v in ds):
        share = as_sparse(share.fillna(0))
    elif any(isinstance(ds[v], COO) for v in ds):
        TypeError('All variables of the dataset must either be sparse or dense.')

    logger.info('Expanding by sink carrier')
    if chunksize is None:
        res = ds * share
    else:
        chunk = {'snapshot': chunksize}
        with ProgressBar():
            res = (ds.chunk(chunk) * share.chunk(chunk)).compute()
    return res.assign_attrs(ds.attrs)
