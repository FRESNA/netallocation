#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 14:13:56 2020

@author: fabian
"""

import pytest
import pypsa
import pandas as pd
import netallocation as ntl
from  netallocation.cost import (nodal_co2_cost, nodal_demand_cost,
                                 nodal_production_revenue, congestion_revenue,
                                 allocate_cost, locational_market_price)
from netallocation.grid import energy_production
from netallocation.utils import get_ext_branches_b, reindex_by_bus_carrier
from xarray.testing import assert_allclose, assert_equal
import xarray as xr
from pypsa.descriptors import nominal_attrs
close = lambda d1, d2: d1.round(0) == d2.round(0)



def check_duality(n):
    O = n.objective + ntl.cost.objective_constant(n)
    PR = nodal_production_revenue(n).sum()
    CO2C = nodal_co2_cost(n).sum()
    DC = nodal_demand_cost(n).sum()
    CR = congestion_revenue(n, split=True).sum()
    assert close(O, PR - CO2C - CR['ext'])
    assert close(O, DC - CO2C + CR['fix'])


def check_zero_profit_branches(n):
    comps = sorted(n.branch_components)
    cost_ext = pd.concat({c: n.df(c)[attr + '_opt'] * n.df(c).capital_cost
                          for c, attr in pypsa.descriptors.nominal_attrs.items()
                          if c in comps})[comps]\
                 .where(get_ext_branches_b(n).to_series(), 0)
    CR = congestion_revenue(n, split=True)['ext'].sum('snapshot')
    assert all(close(cost_ext + CR, xr.zeros_like(CR))), (
        "Zero Profit condition for branches is not fulfilled.")


def check_zero_profit_generators(n):
    cost_inv = n.generators.query('p_nom_extendable')\
                .eval('p_nom_opt * capital_cost')\
                .reindex(n.generators.index, fill_value=0)
    cost_inv = reindex_by_bus_carrier(cost_inv, 'Generator', n).sum('carrier')
    w = n.snapshot_weightings
    cost_op = (n.generators_t.p * n.generators.marginal_cost).mul(w, axis=0).sum()
    cost_op = reindex_by_bus_carrier(cost_op, 'Generator', n).sum('carrier')
    cost = cost_inv + cost_op

    PR = nodal_production_revenue(n, per_carrier=True)
    carriers = [c for c in ntl.utils.generation_carriers(n) if c in PR.carrier]
    PR = PR.sel(carrier=carriers).sum(['carrier', 'snapshot']).reindex_like(cost)
    CO2C = nodal_co2_cost(n).sum('snapshot')
    CO2C = CO2C.reindex_like(cost)

    assert all(close(cost, PR - CO2C)), ("Zero Profit "
            "condition for generators is not fulfilled.")

# reindex_by_bus_carrier(n.generators_t.mu_lower +  n.generators_t.mu_upper,
# 'Generator', n).rename(name='snapshot') - locational_market_price(n)


# =============================================================================
# Test functions
# =============================================================================

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


# TODO: This does only work when the shadow prices of the upper capacity
# generator bound is taken into account.

# def test_duality_wo_investment_2():
#     n = ntl.test.get_network_ac_dc()
#     n.lopf(solver_name='cbc', pyomo=False)
#     for c, attr in pypsa.descriptors.nominal_attrs.items():
#         n.df(c)[attr] = n.df(c)[attr + '_opt'] + 0.01
#         n.df(c)[attr + '_extendable'] = False
#     n.lopf(solver_name='cbc', pyomo=False, keep_shadowprices=True)
#     check_duality(n)
#     check_zero_profit_branches(n)
#     check_zero_profit_generators(n)


def test_duality_with_investment_wo_CO2():
    n = ntl.test.get_network_ac_dc()
    n.generators['p_nom_min'] = 0
    n.remove('GlobalConstraint', 'co2_limit')
    n.lopf(pyomo=False, keep_shadowprices=True, solver_name='cbc')
    check_duality(n, co2_constr_name='co2_limit')
    check_zero_profit_branches(n)
    check_zero_profit_generators(n)


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
    check_zero_profit_generators(n)

# TODO: This does only work when the shadow prices of the upper capacity
# generator bound is taken into account.

# def test_duality_with_investment():
#     n = ntl.test.get_network_ac_dc()
#     n.generators.loc[['Manchester Wind', 'Manchester Gas'],
#                       'p_nom_extendable'] = False
#     n.lopf(pyomo=False, keep_shadowprices=True, solver_name='cbc')
#     check_duality(n, co2_constr_name='co2_limit')
#     check_zero_profit_branches(n)
#     check_zero_profit_generators(n)


def test_duality_investment_mix_ext_nonext_lines():
    n = ntl.test.get_network_ac_dc()
    n.lines.loc['0', 's_nom_extendable'] = False
    n.lopf(pyomo=False, keep_shadowprices=True, solver_name='cbc')
    check_duality(n, co2_constr_name='co2_limit')
    check_zero_profit_branches(n)
    check_zero_profit_generators(n)


