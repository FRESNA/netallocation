#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 17 12:28:37 2018

@author: fabian
"""

import unittest
import pandas as pd
import netallocation as al
from netallocation import allocate_flow
from netallocation.grid import (network_flow, network_injection,
                                self_consumption, Incidence)
from netallocation.utils import get_test_network

n = get_test_network()
snapshot = n.snapshots[0]

#%%

branch_components = ['Line']
f = network_flow(n, snapshot, branch_components)
p = network_injection(n, snapshot, branch_components)
K = Incidence(n, branch_components)
self_con = self_consumption(n, snapshot)


def injection_sum_source(method, branch_components, **kws):
    return allocate_flow(n, snapshot, method=method, per_bus=True,
                branch_components=branch_components, **kws).sum(level=['source'])

def injection_sum_sink(method, branch_components, **kws):
    return allocate_flow(n, snapshot, method=method, per_bus=True,
                branch_components=branch_components, **kws).sum(level=['sink'])

def flow_sum(method, branch_components, **kws):
    return allocate_flow(n, snapshot, method=method,
               branch_components=branch_components, **kws).sum(level=['branch_i'])

class FlowAllocationSubnetTest(unittest.TestCase):

    def test_flow_sum_mp(self):
        res = flow_sum('mp', branch_components)
        self.assertLess((f - res).abs().max(), 1e-4)

    def test_injection_sum_source_mp(self):
        res = injection_sum_source('mp', branch_components)
        self.assertLess((p - res).abs().max(), 1e-4)

    def test_injection_sum_sink_mp(self):
        res = injection_sum_sink('mp', branch_components) # has to be zero
        self.assertLess((res).abs().max(), 1e-4)

#%%
    def test_flow_sum_ap(self):
        res = flow_sum('ap', branch_components)
        self.assertLess((f - res).abs().max(), 1e-4)

    def test_injection_sum_source_ap(self):
        res = injection_sum_source('ap', branch_components)
        self.assertLess((self_con + p.clip(lower=0) - res).abs().max(), 1e-4)

    def test_injection_sum_sink_ap(self):
        res = injection_sum_sink('ap', branch_components) # has to be zero
        self.assertLess((p.clip(upper=0) + res - self_con).abs().max(), 1e-4)

#%%
    def test_flow_sum_vip(self):
        res = flow_sum('vip', branch_components)
        self.assertLess((f - res).abs().max(), 1e-4)

    def test_injection_sum_source_vip(self):
        res = injection_sum_source('vip', branch_components)
        self.assertLess((p[p>0] - res).abs().max(), 1e-4)

    def test_injection_sum_sink_vip(self):
        res = injection_sum_sink('vip', branch_components) # has to be zero
        self.assertLess((p[p<0] + res).abs().max(), 1e-4)

#%%
    def test_flow_sum_zbus(self):
        res = flow_sum('zbus', branch_components, linear=True)
        self.assertLess((f - res).abs().max(), 1e-4)

#    def test_injection_sum_source_vip(self):
#        res = injection_sum_source('vip', branch_components)
#        self.assertLess((p[p>0] - res).abs().max(), 1e-4)
#
#    def test_injection_sum_sink_vip(self):
#        res = injection_sum_sink('vip', branch_components) # has to be zero
#        self.assertLess((p[p<0] + res).abs().max(), 1e-4)



#%%

print('\n Test incorparation of links with pseudo impedance:')

n = get_test_network()
snapshot = n.snapshots[0]
branch_components = ['Line', 'Link']
f = network_flow(n, snapshot, branch_components)
p = network_injection(n, snapshot, branch_components)
K = Incidence(n, branch_components)
self_con = self_consumption(n, snapshot)

class FlowAllocationSupernetTest(unittest.TestCase):


    def test_flow_sum_mp(self):
        res = flow_sum('mp', branch_components)
        self.assertLess((f - res).abs().max(), 1e-4)

    def test_injection_sum_source_mp(self):
        res = injection_sum_source('mp', branch_components)
        self.assertLess((p - res).abs().max(), 1e-4)

    def test_injection_sum_sink_mp(self):
        res = injection_sum_sink('mp', branch_components) # has to be zero
        self.assertLess((res).abs().max(), 1e-4)

#------------------------------------------------
    def test_flow_sum_ap(self):
        res = flow_sum('ap', branch_components)
        self.assertLess((f - res).abs().max(), 1e-4)

    def test_injection_sum_source_ap(self):
        res = injection_sum_source('ap', branch_components)
        self.assertLess((self_con + p.clip(lower=0) - res).abs().max(), 1e-4)

    def test_injection_sum_sink_ap(self):
        res = injection_sum_sink('ap', branch_components) # has to be zero
        self.assertLess((p.clip(upper=0) + res - self_con).abs().max(), 1e-4)

#-------------------------------------------------
    def test_flow_sum_vip(self):
        res = flow_sum('vip', branch_components)
        self.assertLess((f - res).abs().max(), 1e-4)

    def test_injection_sum_source_vip(self):
        res = injection_sum_source('vip', branch_components)
        self.assertLess((p[p>0] - res).abs().max(), 1e-4)

    def test_injection_sum_sink_vip(self):
        res = injection_sum_sink('vip', branch_components) # has to be zero
        self.assertLess((p[p<0] + res).abs().max(), 1e-4)

unittest.main()
