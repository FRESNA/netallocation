#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 14:43:28 2019

@author: fabian
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 17 12:28:37 2018

@author: fabian
"""

import unittest
import netallocation as al
from netallocation import allocate_flow
from netallocation.grid import (network_flow, network_injection,
                                self_consumption, Incidence)


n = al.utils.get_test_network(linear=False)
branch_components = ['Line', 'Transformer']
snapshot = n.snapshots[0]
f = (network_flow(n, snapshot, branch_components) \
    - network_flow(n, snapshot, branch_components, ingoing=False))/2

#%%
print('\n Test allocation flor non-linear power flow:')

def injection_sum_source(method, branch_components):
    return allocate_flow(n, snapshot, method=method, per_bus=True,
                branch_components=branch_components).sum(level=['source'])

def injection_sum_sink(method, branch_components):
    return allocate_flow(n, snapshot, method=method, per_bus=True,
                branch_components=branch_components).sum(level=['sink'])

def flow_sum(method, branch_components):
    return allocate_flow(n, snapshot, method=method,
               branch_components=branch_components)\
                         .sum(level=['component', 'branch_i'])


class FlowAllocationComplexTest(unittest.TestCase):


    def test_flow_sum_mp(self):
        res = flow_sum('zbus', branch_components)
        self.assertLess((f - res).abs().max(), 1e-4)

unittest.main()
