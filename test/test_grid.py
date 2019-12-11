#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 11:21:05 2019

@author: fabian
"""

#import unittest
from netallocation.grid import network_injection, Incidence, network_flow
import netallocation as ntl
import pypsa
from pathlib import Path
from numpy.testing import assert_allclose, assert_array_equal
import os

n = pypsa.Network(os.path.join(__file__, '..',  'test.nc'))
n_dc = pypsa.Network(os.path.join(__file__, '..',  'simple_dc_model.nc'))
n_large = pypsa.Network(os.path.join(__file__, '..',  'european_model.nc'))
tol_kwargs = dict(atol=1e-5, rtol=1)

def test_injection():
    assert_allclose(
            network_injection(n, branch_components=n.passive_branch_components).values,
            n.buses_t.p.values, **tol_kwargs)

def test_branch_order():
    assert_array_equal(Incidence(n).get_index('branch'),
            network_flow(n, snapshots=n.snapshots[0]).get_index('branch'))

    pbc = n.passive_branch_components
    assert_array_equal(Incidence(n, branch_components=pbc).get_index('branch'),
            network_flow(n, snapshots=n.snapshots[0], branch_components=pbc)
            .get_index('branch'))

def test_cycles():
    C = ntl.grid.Cycles(n)
    pypsa.pf.find_cycles(n)
    assert_array_equal(C.values, n.C.todense())

    C = ntl.grid.Cycles(n_large)
    pypsa.pf.find_cycles(n_large)
    assert_array_equal(C.values, n_large.C.todense())


def test_pseudo_impedance_ac_dc():
    sn = n.snapshots[0]
    H = ntl.grid.PTDF(n, branch_components=n.branch_components, snapshot=sn)
    p = ntl.grid.network_injection(n, sn)
    f = ntl.grid.network_flow(n, sn)
    assert_allclose(H @ p, f, **tol_kwargs)

def test_pseudo_impedance_dc():
    sn = n_dc.snapshots[0]
    H = ntl.grid.PTDF(n_dc, branch_components=n.branch_components, snapshot=sn)
    p = ntl.grid.network_injection(n_dc, sn)
    f = ntl.grid.network_flow(n_dc, sn)
    assert_allclose(H @ p, f, **tol_kwargs)

def test_pseudo_impedance_ac_dc_large():
    sn = n_large.snapshots[0]
    H = ntl.grid.PTDF(n_large, branch_components=n.branch_components, snapshot=sn)
    p = ntl.grid.network_injection(n_large, sn)
    f = ntl.grid.network_flow(n_large, sn)
    assert_allclose(H @ p, f, **tol_kwargs)


if __name__ == '__main__':
    test_injection()
    test_branch_order()
    test_cycles()
    test_pseudo_impedance_ac_dc()
    test_pseudo_impedance_dc()
    test_pseudo_impedance_ac_dc_large()
