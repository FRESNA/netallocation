#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 15:38:24 2019

@author: fabian
"""

from .grid import power_demand, power_production
from .utils import as_sparse, obj_if_acc
import logging

logger = logging.getLogger(__name__)

def expand_by_source_type(ds, n, components=['Generator', 'StorageUnit'],
                          dim='source', cut_lower_share=1e-5, sparse=True,
                          chunk=None):
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


    Example
    -------
    >>> ap = ntl.flow_allocation(n, n.snapshots, method='ap')
    >>> ntl.breakdown.expand_by_source_carrier(ap, n)

    """
    ds = obj_if_acc(ds)
    sns = ds.get_index('snapshot')
    share = (power_production(n, sns, per_carrier=True) / power_production(n, sns))
    share = share.rename(bus='source', carrier='source_carrier')
    if sparse:
        share = as_sparse(share.fillna(0))
    logger.info('Expanding by source carrier')
    if chunk is None:
        return (ds * share).assign_attrs(ds.attrs)
    else:
        return (ds.chunk(chunk) * share.chunk(chunk)).compute().assign_attrs(ds.attrs)
            #.stack({'production': ('source', 'source_carrier')})


def expand_by_sink_type(ds, n, components=['Load', 'StorageUnit'],
                        cut_lower_share=1e-5, sparse=True):
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

    Example
    -------
    >>> ap = ntl.flow_allocation(n, n.snapshots, method='ap')
    >>> ntl.breakdown.expand_by_sink_carrier(ap, n)

    """
    ds = obj_if_acc(ds)
    sns = ds.get_index('snapshot')
    share = (power_demand(n, sns, per_carrier=True) / power_demand(n, sns))
    share = share.rename(bus='sink', carrier='sink_carrier')
    if sparse:
        share = as_sparse(share.fillna(0))
    logger.info('Expanding by sink carrier')
    return (ds * share).assign_attrs(ds.attrs)
            #.stack({'demand': ('sink', 'sink_carrier')})
