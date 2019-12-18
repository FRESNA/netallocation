#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 10:16:50 2019

@author: fabian
"""

from .flow import flow_allocation as allocate_flow
from .grid import (Incidence, network_flow, network_injection,
                   power_demand, power_production)
from .linalg import diag, inv, pinv

from . import utils, breakdown, grid, flow, linalg, plot, cost, plot_helpers, \
                test, io, common

#pd.Series.__rshift__ = lambda d1, d2: pd.testing.assert_series_equal(
#        d1.round(6), d2.round(6).reindex_like(d1), check_exact=False, check_names=False)
#
#pd.DataFrame.__rshift__ = lambda d1, d2: pd.testing.assert_frame_equal(
#        d1.round(6), d2.round(6).reindex_like(d1), check_exact=False, check_names=False)
