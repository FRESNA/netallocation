#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 14:02:51 2019

@author: fabian
"""


from .flow import flow_allocation
from .breakdown import expand_by_source_type, expand_by_sink_type
import pandas as pd
import logging


class Allocator:
    def __init__(self, network, snapshots=None, method = 'Average participation',
                 flow_dimensions=['source', 'sink']):
        self.network = network
        self.snapshots = network.snapshots if snapshots is None else snapshots
        self.method = method
        self.flow_dimensions = flow_dimensions
        self._calculated = False
        logging.info(f'Initialize Allocator for allocating on dimension '
                     f'{flow_dimensions} with method {method}')

    def __repr__(self):
        info_dict = {'Network': self.network,
                     'Number of snapshots': len(self.snapshots),
                     'method': self.method,
                     'Dimension': ', '.join(self.flow_dimensions),
                     'calculated': self._calculated}
        return f'<netallocation.Allocator> \n {info_dict}'

    def set_method(self, method):
        self.method = method
        return self


    def set_flow_dimensions(self, dims):
        self.flow_dimensions = dims
        return self


    def calculate(self):
        per_bus = 'branch' not in self.flow_dimensions
        ds = flow_allocation(self.network, self.snapshots, self.method,
                             per_bus=per_bus)
        if 'sourcetype' in self.flow_dimensions:
            ds = expand_by_source_type(ds, self.network)
        if 'sinktype' in self.flow_dimensions:
            ds = expand_by_sink_type(ds, self.network)

        setattr(self, '_data_' + '_'.join(sorted(self.flow_dimensions)), ds)
        self._calculated = True
        return self

    def get_data(self):
        if not hasattr(self, '_data_' + '_'.join(sorted(self.flow_dimensions))):
            self.calculate()
        return getattr(self, '_data_' + '_'.join(sorted(self.flow_dimensions)))




#
#    def source_to_sink(self, **kwargs):
#        kwargs['per_bus'] = True
#
#        self.data_source_to_sink = flow_allocation(self.network, self.snapshots,
#                                                    method=method, **kwargs)
#
#    def source_to_sink_on_branch(self, method='Average participation', **kwargs):
#        kwargs['per_bus'] = False
#
#        self.data_source_to_sink_on_branch = flow_allocation(self.network, self.snapshots,
#                                                    method=method, **kwargs)
#

