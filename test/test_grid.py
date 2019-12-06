#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 11:21:05 2019

@author: fabian
"""

#import unittest
from netallocation.grid import network_injection, Incidence, network_flow
import pypsa
#from pathlib import Path
#from numpy import allclose
from numpy.testing import assert_allclose, assert_array_equal

n = pypsa.Network('test/test.nc')

def test_injection():
    assert_allclose(
            network_injection(n, branch_components=n.passive_branch_components).values,
            n.buses_t.p.values, 1e-5, 1)

def test_branch_order():
    assert_array_equal(Incidence(n).get_index('branch'),
            network_flow(n, snapshots=n.snapshots[0]).get_index('branch'))

    pbc = n.passive_branch_components
    assert_array_equal(Incidence(n, branch_components=pbc).get_index('branch'),
            network_flow(n, snapshots=n.snapshots[0], branch_components=pbc)
            .get_index('branch'))


if __name__ == '__main__':
    test_injection()
    test_branch_order()