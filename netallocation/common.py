#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 17:46:45 2019

@author: fabian
"""

import xarray as xr


@xr.register_dataset_accessor("ntl")
class AllocationAccessor:
    """
    Accessor for netallocation package.

    This accessor enables to call several functions to modify, plot and store
    the allocation dataset.

    """

    def __init__(self, xarray_obj):
        self._obj = xarray_obj

    from .io import store_dataset as to_dir
    from .breakdown import expand_by_sink_type, expand_by_source_type
    from .utils import as_dense, as_sparse, filter_null, lower, upper
    from .plot import chord_diagram as plot_chord_diagram

