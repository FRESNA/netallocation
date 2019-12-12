#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 11:21:05 2019

@author: fabian
"""

#import unittest
from netallocation.grid import network_injection, Incidence, network_flow
import netallocation as ntl
import xarray as xr
import pypsa
from numpy.testing import assert_allclose, assert_array_equal
import numpy as np

n = ntl.test.get_network_ac_dc()
n_dc = ntl.test.get_network_pure_dc_link()
n_large = ntl.test.get_network_large()

tol_kwargs = dict(atol=1e-5, rtol=1)

def test_injection():
    comps = n.passive_branch_components
    assert_allclose(network_injection(n, branch_components=comps).values,
                    n.buses_t.p.values, **tol_kwargs)

def test_branch_order():
    assert_array_equal(Incidence(n).get_index('branch'),
            network_flow(n, snapshots=n.snapshots[0]).get_index('branch'))

    pbc = n.passive_branch_components
    assert_array_equal(Incidence(n, branch_components=pbc).get_index('branch'),
            network_flow(n, snapshots=n.snapshots[0], branch_components=pbc)
            .get_index('branch'))

def test_bus_order():
    comps = n.passive_branch_components
    assert network_injection(n, branch_components=comps).get_index('bus')\
           .equals(Incidence(n, branch_components=comps).get_index('bus'))

def test_cycles():
    # since cycles are degenerated use single detection
    C = ntl.grid.Cycles(n)
    pypsa.pf.find_cycles(n)
    C2 = xr.DataArray(n.C.todense(), {'branch': n.branches().index,
               'cycle': range(C.shape[1])}, ['branch', 'cycle']).reindex_like(C)
    Cp = C.to_pandas().T
    def cycle_in_C(cycle):
        return (cycle == Cp).all(1).any() or (cycle == - Cp).all(1).any()
    assert C2.to_pandas().apply(cycle_in_C).all()

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

