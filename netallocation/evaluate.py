#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 13:23:56 2020

@author: fabian
"""

from .convert import virtual_patterns
from .utils import check_snapshots
from .grid import network_flow, network_injection
import numpy as np


# 3a)
def gross_network_use(ds):
    da = ds.virtual_flow_pattern
    return (da.sum('branch') / da.sum(['bus', 'branch'])).sum('snapshot')


# 3b)
def network_use_or_stress(ds, n):
    da = ds.virtual_flow_pattern
    f = network_flow(n, da.snapshot)
    return (np.sign(f) * da).sum('branch') / abs(f).sum('branch')
    # aligned = da.where(da * np.sign(f) >= 0, 0).sum('branch')
    # counter = da.where(da * np.sign(f) <= 0, 0).sum('branch')
    # return (abs(aligned) - abs(counter)) / abs(f).sum('branch')


def trade_dependency(n, snapshots):
    snapshots = check_snapshots(snapshots, n)
    trade = abs(network_injection(n, snapshots)).sum('snapshot')
    return trade / trade.sum('bus')
