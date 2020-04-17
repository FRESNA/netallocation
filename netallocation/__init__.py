#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This is the netallocation package.

It package provides various functions to allocate flow in a pypsa power
system. Underlying packages are xarray and dask.

"""

from .flow import flow_allocation as allocate_flow
from .grid import (Incidence, network_flow, network_injection,
                   power_demand, power_production)
from .linalg import diag, inv, pinv
from .io import load_dataset, store_dataset
from .utils import as_dense, as_sparse
from .cost import allocate_cost

from . import (utils, breakdown, grid, flow, linalg, plot, cost,
               plot_helpers, test, io, common, process, evaluate)

__version__ = '0.0.4'
__author__ = "Fabian Hofmann (FIAS)"
__copyright__ = "Copyright 2015-2020 Fabian Hofmann (FIAS), GNU GPL 3"
