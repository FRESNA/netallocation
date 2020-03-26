#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 14:13:56 2020

@author: fabian
"""

import pypsa
import pandas as pd
import netallocation as ntl
from  netallocation.cost import (nodal_co2_cost, nodal_demand_cost,
                                 nodal_production_revenue, congestion_revenue,
                                 weight_with_generation_cost, allocate_cost)
from netallocation.utils import get_ext_branches_i, reindex_by_bus_carrier
from xarray.testing import assert_allclose, assert_equal
from pypsa.descriptors import nominal_attrs
close = lambda d1, d2: d1.round(0) == d2.round(0)


def check_duality(n, co2_constr_name=None):
    O = n.objective + ntl.cost.objective_constant(n)
    PR = nodal_production_revenue(n).sum()
    CO2C = nodal_co2_cost(n).sum()
    DC = nodal_demand_cost(n).sum()
    CR_ext, CR_fix = congestion_revenue(n, split=True)
    assert close(O, PR - CO2C - CR_ext.sum())
    assert close(O, DC - CO2C + CR_fix.sum())

def check_zero_profit_branches(n):
    cost_ext = n.branches().fillna(0)\
                .eval('(s_nom_opt + p_nom_opt) * capital_cost')\
                [get_ext_branches_i(n)]
    CR_ext, CR_fix = congestion_revenue(n, split=True)
    assert all(close(cost_ext, -CR_ext.sum('snapshot'))), ("Zero Profit "
            "condition for branches is not fulfilled.")


def check_zero_profit_generators(n):
    cost_ext = n.generators.query('p_nom_extendable')\
                .eval('p_nom_opt * capital_cost')
    cost_ext = reindex_by_bus_carrier(cost_ext, 'Generator', n).sum('carrier')
    PR_ext, _ = nodal_production_revenue(n, split=True)
    PR_ext = PR_ext.sum('snapshot').reindex_like(cost_ext)
    assert all(close(cost_ext, PR_ext)), ("Zero Profit "
            "condition for generators is not fulfilled.")


def test_duality_wo_investment():
    n = ntl.test.get_network_mini()
    for c, attr in nominal_attrs.items():
        n.df(c)[attr + '_extendable'] = False
    n.lopf(pyomo=False, keep_shadowprices=True, solver_name='cbc')
    check_duality(n)
    check_zero_profit_branches(n)


def test_duality_wo_investment_sn_weightings():
    n = ntl.test.get_network_mini()
    for c, attr in nominal_attrs.items():
        n.df(c)[attr + '_extendable'] = False
    n.snapshot_weightings[:] = 3
    n.lopf(pyomo=False, keep_shadowprices=True, solver_name='cbc')
    check_duality(n)
    check_zero_profit_branches(n)


# TODO
def test_duality_wo_investment_2():
    n = ntl.test.get_network_ac_dc()
    n.lopf(solver_name='cbc')
    for c, attr in pypsa.descriptors.nominal_attrs.items():
        n.df(c)[attr] = n.df(c)[attr + '_opt']
        n.df(c)[attr + '_extendable'] = False
    n.lopf(solver_name='cbc')
    check_duality(n)
    check_zero_profit_branches(n)

# TODO
def test_duality_with_investment_wo_CO2():
    n = ntl.test.get_network_ac_dc()
    n.remove('GlobalConstraint', 'co2_limit')
    n.lopf(pyomo=False, keep_shadowprices=True, solver_name='cbc')
    check_duality(n, co2_constr_name='co2_limit')
    check_zero_profit_branches(n)


def test_duality_with_investment():
    n = ntl.test.get_network_ac_dc()
    for c, attr in pypsa.descriptors.nominal_attrs.items():
        n.df(c)[attr] = 0
    n.lopf(pyomo=False, keep_shadowprices=True, solver_name='cbc')
    check_duality(n, co2_constr_name='co2_limit')
    check_zero_profit_branches(n)
    check_zero_profit_generators(n)


def test_duality_with_investment_sn_weightings():
    n = ntl.test.get_network_ac_dc()
    n.snapshot_weightings[:] = 3
    n.global_constraints.constant *= 3
    n.lopf(pyomo=False, keep_shadowprices=True, solver_name='cbc')
    check_duality(n, co2_constr_name='co2_limit')
    check_zero_profit_branches(n)


# def test_duality_with_investment():
#     n = ntl.test.get_network_ac_dc()
#     n.generators.loc[['Manchester Wind', 'Manchester Gas'],
#                       'p_nom_extendable'] = False
#     n.lopf(pyomo=False, keep_shadowprices=True, solver_name='cbc')
#     check_duality(n, co2_constr_name='co2_limit')
#     check_zero_profit_branches(n)


def test_duality_investment_mix_ext_nonext_lines():
    n = ntl.test.get_network_ac_dc()
    n.lines.loc['0', 's_nom_extendable'] = False
    n.lopf(pyomo=False, keep_shadowprices=True, solver_name='cbc')
    check_duality(n, co2_constr_name='co2_limit')
    check_zero_profit_branches(n)


def test_cost_allocation_sum():
    n = ntl.test.get_network_ac_dc()
    total_allocated_cost = allocate_cost(n).sum('sink').rename(source='bus')
    p = ntl.power_production(n)
    total_production_cost = weight_with_generation_cost(p, n, dim='bus')\
                                .sum('carrier')
    assert_allclose(total_allocated_cost, total_production_cost)

