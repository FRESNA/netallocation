#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 15:38:24 2019

@author: fabian
"""

from .grid import self_consumption, power_demand, power_production, network_flow
from .utils import as_sparse
import pandas as pd


def expand_by_source_type(ds, n, components=['Generator', 'StorageUnit'],
                          dim='source', cut_lower_share=1e-5, sparse=True):
    """
    Breakdown allocation into generation carrier type. These include carriers
    of all components specified by 'components'. Note that carrier names of all
    components have to be unique.

    Pararmeter
    ----------

    ds : pd.Series
        Allocation Series with at least index level 'source'
    n : pypsa.Network()
        Network which the allocation was derived from
    components : list, default ['Generator', 'StorageUnit']
        List of considered components. Carrier types of these components are
        taken for breakdown.


    Example
    -------

    ap = flow_allocation(n, n.snapshots, per_bus=True)
    ap_carrier = pypsa.allocation.expand_by_carrier(ap, n)

    """
    sns = ds.get_index('snapshot')
    share = (power_production(n, sns, per_carrier=True) / power_production(n, sns))
    share = share.rename(bus='source', carrier='source_carrier')
    if sparse:
        share = as_sparse(share.fillna(0))
    return (ds * share)#.stack({'production': ('source', 'source_carrier')})



def expand_by_sink_type(ds, n, components=['Load', 'StorageUnit'],
                        cut_lower_share=1e-5, sparse=True):
    """
    Breakdown allocation into demand types, e.g. Storage carriers and Load.
    These include carriers of all components specified by 'components'. Note
    that carrier names of all components have to be unique.

    Pararmeter
    ----------

    ds : pd.Series
        Allocation Series with at least index level 'sink'
    n : pypsa.Network()
        Network which the allocation was derived from
    components : list, default ['Load', 'StorageUnit']
        List of considered components. Carrier types of these components are
        taken for breakdown.


    Example
    -------

    ap = flow_allocation(n, n.snapshots, per_bus=True)
    ap_carrier = pypsa.allocation.expand_by_carrier(ap, n)

    """
    sns = ds.get_index('snapshot')
    share = (power_demand(n, sns, per_carrier=True) / power_demand(n, sns))
    share = share.rename(bus='sink', carrier='sink_carrier')
    if sparse:
        share = as_sparse(share.fillna(0))
    return (ds * share)#.stack({'demand': ('sink', 'sink_carrier')})
