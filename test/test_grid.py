#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 11:21:05 2019

@author: fabian
"""

#import unittest
from netallocation.grid import (network_injection, Incidence, network_flow,
                                power_production)
from netallocation.utils import as_dense
import netallocation as ntl
import xarray as xr
import pypsa
from xarray.testing import assert_allclose, assert_equal
from numpy.testing import assert_allclose as np_assert_allclose, \
                        assert_array_equal as np_assert_equal


n = ntl.test.get_network_ac_dc()
n_dc = ntl.test.get_network_pure_dc_link()
n_large = ntl.test.get_network_large()
n_mini = ntl.test.get_network_mini()
tol_kwargs = dict(atol=1e-5, rtol=1)


def test_injection():
    comps = n.passive_branch_components
    np_assert_allclose(network_injection(n, branch_components=comps).values,
                    n.buses_t.p.values, **tol_kwargs)

def test_branch_order():
    np_assert_equal (Incidence(n).get_index('branch'),
            network_flow(n, snapshots=n.snapshots[0]).get_index('branch'))

    pbc = n.passive_branch_components
    np_assert_equal (Incidence(n, branch_components=pbc).get_index('branch'),
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


def test_ordinary_PTDF():
    p = ntl.network_injection(n, branch_components=n.passive_branch_components)
    f = ntl.network_flow(n, branch_components=n.passive_branch_components)
    H = ntl.grid.PTDF(n, branch_components=n.passive_branch_components)
    assert_allclose(H @ p, f, **tol_kwargs)

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

def test_average_participation_aggregated():
    sn = n.snapshots[0]
    K = ntl.Incidence(n)
    A = ntl.flow.flow_allocation(n, sn, method='ap')
    #check total injection
    total_injection = A.peer_to_peer.sum('sink')
    target = power_production(n, sn).rename(bus='source')
    assert_allclose(total_injection, target, **tol_kwargs)

    #check total flow for peer_on_branch_to_peer
    total_flow = A.peer_on_branch_to_peer.sum(['source', 'sink'])
    assert_allclose(total_flow, network_flow(n, sn), **tol_kwargs)
    #check ingoing flow for peer_on_branch_to_peer
    sinkflow = A.peer_on_branch_to_peer.sum('source')
    assert_allclose((sinkflow @ K).sum('sink'), network_injection(n, sn), **tol_kwargs)

def test_average_participation_direct():
    sn = n.snapshots[0]
    K = ntl.Incidence(n)
    A = ntl.flow.flow_allocation(n, sn, method='ap', aggregated=False)
    #check total injection
    total_injection = A.peer_to_peer.sum('sink')
    target = power_production(n, sn).rename(bus='source')
    assert_allclose(total_injection, target, **tol_kwargs)

    #check total flow for peer_on_branch_to_peer
    total_flow = A.peer_on_branch_to_peer.sum(['source', 'sink'])
    assert_allclose(total_flow, network_flow(n, sn), **tol_kwargs)
    #check ingoing flow for peer_on_branch_to_peer
    sinkflow = A.peer_on_branch_to_peer.sum('source')
    assert_allclose((sinkflow @ K).sum('sink'), network_injection(n, sn), **tol_kwargs)



def test_marginal_participation():
    sn = n.snapshots[0]
    A = ntl.flow.flow_allocation(n, sn, method='mp')
    total_injection = A.virtual_injection_pattern.sum('injection_pattern')
    assert_allclose(total_injection, network_injection(n, sn), **tol_kwargs)
    total_flow = A.virtual_flow_pattern.sum('bus')
    assert_allclose(total_flow, network_flow(n, sn), **tol_kwargs)

def test_eqivalent_bilateral_exchanges():
    sn = n.snapshots[0]
    A = ntl.flow.flow_allocation(n, sn, method='ebe')
    total_injection = A.virtual_injection_pattern.sum('injection_pattern')
    assert_allclose(total_injection, network_injection(n, sn), **tol_kwargs)
    total_flow = A.virtual_flow_pattern.sum('bus')
    assert_allclose(total_flow, network_flow(n, sn), **tol_kwargs)

def test_zbus_linear():
    sn = n_mini.snapshots
    A = ntl.flow.flow_allocation(n_mini, sn, method='zbus')
    total_injection = A.virtual_injection_pattern.sum('injection_pattern')
    assert_allclose(total_injection, network_injection(n_mini, sn), **tol_kwargs)
    total_flow = A.virtual_flow_pattern.sum('bus')
    assert_allclose(total_flow, network_flow(n_mini, sn).T, **tol_kwargs)

def test_average_participation_sparse():
    sn = n.snapshots[0]
    A_sp = ntl.flow.flow_allocation(n, sn, method='ap', dims='all', sparse=True)
    A = ntl.flow.flow_allocation(n, sn, method='ap', dims='all', sparse=False)
    assert_allclose(as_dense(A_sp), A, **tol_kwargs)

